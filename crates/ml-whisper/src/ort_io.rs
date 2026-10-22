//! The two places this crate touches ONNX Runtime's tensors: the encoder run,
//! and the decoder run behind [`crate::cached::DecoderRun`].
//!
//! Everything above these functions works on plain `Vec<f32>`; everything
//! that needs the runtime is here, so it is also everything CI cannot run
//! without a model. `tests/transcribe_known_words.rs` runs it.
//!
//! The sessions are built through `ml_pipeline::ModelRegistry`, which calls
//! `ml_pipeline::runtime::ensure()` first; nothing in this crate lets `ort`
//! load the library on its own (it panics when that fails, CLAUDE.md).

use std::sync::MutexGuard;

use ml_pipeline::SharedSession;
use ort::session::Session;
use ort::value::{Outlet, Tensor, TensorElementType, TensorRef};

use crate::cached::{DecoderRun, Feed, Fetched, Want};
use crate::decode::{DecodeError, EncoderStates};
use crate::export::{cross_name, past_name, present_name, Elem, ExportLayout, TensorSpec};
use crate::mel::{N_FRAMES, N_MELS};
use crate::WhisperError;

/// A session's inputs or outputs as names, element types and shapes.
pub(crate) fn specs(outlets: &[Outlet]) -> Vec<TensorSpec> {
    outlets
        .iter()
        .map(|o| TensorSpec {
            name: o.name().to_string(),
            elem: match o.dtype().tensor_type() {
                Some(TensorElementType::Float32) => Elem::F32,
                Some(TensorElementType::Int64) => Elem::I64,
                Some(TensorElementType::Bool) => Elem::Bool,
                _ => Elem::Other,
            },
            dims: o
                .dtype()
                .tensor_shape()
                .map(|s| s.to_vec())
                .unwrap_or_default(),
        })
        .collect()
}

fn inference(what: &str, e: impl std::fmt::Display) -> WhisperError {
    WhisperError::Inference(format!("{what}: {e}"))
}

/// Run the encoder on one window's log-mel (`[80, 3000]`, row-major, as
/// `LogMel::into_vec` gives it) and return its output.
pub(crate) fn run_encoder(
    session: &SharedSession,
    mel: Vec<f32>,
    layout: &ExportLayout,
) -> Result<EncoderStates, WhisperError> {
    debug_assert_eq!(mel.len(), N_MELS * N_FRAMES);
    let input = Tensor::from_array(([1usize, N_MELS, N_FRAMES], mel))
        .map_err(|e| inference("encoder input", e))?;
    let mut session = session
        .lock()
        .map_err(|_| WhisperError::Inference("the encoder session's lock is poisoned".into()))?;
    let outputs = session
        .run(ort::inputs!["input_features" => input])
        .map_err(|e| inference("encoder run", e))?;
    let (shape, data) = outputs["last_hidden_state"]
        .try_extract_tensor::<f32>()
        .map_err(|e| inference("encoder output", e))?;
    if shape.len() != 3 || shape[1] != layout.frames as i64 || shape[2] != layout.width as i64 {
        return Err(WhisperError::Inference(format!(
            "the encoder returned {:?}, expected [1, {}, {}]",
            &shape[..],
            layout.frames,
            layout.width
        )));
    }
    Ok(EncoderStates::new(
        data.to_vec(),
        layout.frames,
        layout.width,
    )?)
}

/// The decoder session, locked for the life of one window.
pub(crate) struct OrtDecoder<'s> {
    session: MutexGuard<'s, Session>,
    layout: ExportLayout,
    /// Per layer: `past_key_values.{i}.decoder.key`, `.decoder.value`,
    /// `.encoder.key`, `.encoder.value`.
    past: Vec<[String; 4]>,
    /// The same for `present.{i}...`.
    present: Vec<[String; 4]>,
}

impl<'s> OrtDecoder<'s> {
    pub(crate) fn lock(
        session: &'s SharedSession,
        layout: ExportLayout,
    ) -> Result<Self, WhisperError> {
        let session = session.lock().map_err(|_| {
            WhisperError::Inference("the decoder session's lock is poisoned".into())
        })?;
        let names = |f: fn(usize, bool, bool) -> String| -> Vec<[String; 4]> {
            (0..layout.layers)
                .map(|l| {
                    [
                        f(l, false, false),
                        f(l, false, true),
                        f(l, true, false),
                        f(l, true, true),
                    ]
                })
                .collect()
        };
        Ok(Self {
            session,
            layout,
            past: names(past_name),
            present: names(present_name),
        })
    }
}

fn backend(what: &str, e: impl std::fmt::Display) -> DecodeError {
    DecodeError::Backend(format!("{what}: {e}"))
}

impl DecoderRun for OrtDecoder<'_> {
    fn run(&mut self, feed: &Feed<'_>, want: &Want<'_>) -> Result<Fetched, DecodeError> {
        let l = self.layout;
        let seq = feed.input_ids.len();
        let dec_shape = [1usize, l.heads, feed.past_decoder_len, l.head_dim];
        let enc_shape = [1usize, l.heads, feed.past_encoder_len, l.head_dim];

        let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(3 + 4 * l.layers);
        inputs.push((
            "input_ids".into(),
            Tensor::from_array(([1usize, seq], feed.input_ids.to_vec()))
                .map_err(|e| backend("input_ids", e))?
                .into(),
        ));
        inputs.push((
            "encoder_hidden_states".into(),
            TensorRef::from_array_view((
                [1usize, feed.encoder.frames(), feed.encoder.width()],
                feed.encoder.as_slice(),
            ))
            .map_err(|e| backend("encoder_hidden_states", e))?
            .into(),
        ));
        for (names, layer) in self.past.iter().zip(feed.past) {
            for (name, shape, data) in [
                (&names[0], dec_shape, &layer.dec_key),
                (&names[1], dec_shape, &layer.dec_value),
                (&names[2], enc_shape, &layer.enc_key),
                (&names[3], enc_shape, &layer.enc_value),
            ] {
                inputs.push((
                    name.clone(),
                    TensorRef::from_array_view((shape, &data[..]))
                        .map_err(|e| backend(name, e))?
                        .into(),
                ));
            }
        }
        inputs.push((
            "use_cache_branch".into(),
            Tensor::from_array(([1usize], vec![feed.use_cache]))
                .map_err(|e| backend("use_cache_branch", e))?
                .into(),
        ));

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| backend("decoder run", e))?;

        let extract = |name: &str| -> Result<(&[i64], &[f32]), DecodeError> {
            let value = outputs
                .get(name)
                .ok_or_else(|| DecodeError::Backend(format!("the decoder returned no `{name}`")))?;
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| backend(name, e))?;
            Ok((&shape[..], data))
        };
        // The sequence axis of a `[1, heads, len, head_dim]` tensor.
        let len_of = |dims: &[i64]| dims.get(2).copied().unwrap_or(0).max(0) as usize;

        let logits = extract("logits")?.1.to_vec();

        let mut decoder = Vec::with_capacity(l.layers);
        let mut decoder_len = 0;
        for names in &self.present {
            let (dims, key) = extract(&names[0])?;
            decoder_len = len_of(dims);
            decoder.push((key.to_vec(), extract(&names[1])?.1.to_vec()));
        }
        let (mut encoder, mut encoder_len) = (None, 0);
        if want.encoder_present {
            let mut all = Vec::with_capacity(l.layers);
            for names in &self.present {
                let (dims, key) = extract(&names[2])?;
                encoder_len = len_of(dims);
                all.push((key.to_vec(), extract(&names[3])?.1.to_vec()));
            }
            encoder = Some(all);
        }
        let mut cross = Vec::with_capacity(want.cross_layers.len());
        for &layer in want.cross_layers {
            cross.push(extract(&cross_name(layer))?.1.to_vec());
        }
        Ok(Fetched {
            logits,
            decoder,
            decoder_len,
            encoder,
            encoder_len,
            cross,
        })
    }
}
