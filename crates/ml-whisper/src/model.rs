//! The loaded model, and transcribing with it.
//!
//! [`WhisperModel::load`] takes the folder of the pinned export
//! (`ml_pipeline::models::whisper_base()`: `onnx/encoder_model.onnx`,
//! `onnx/decoder_model_merged_int8.onnx`, `tokenizer.json`), builds both
//! sessions and checks them by name ([`crate::export`]). [`transcribe`] then
//! turns any audio buffer into timed words:
//!
//! 1. [`resample_to_16khz_mono`]: any rate, any channel count.
//! 2. [`crate::chunk::run_windows`]: 30 s windows and where each one starts.
//! 3. For each window: log-mel (`MelExtractor::log_mel_window`, the Hugging
//!    Face convention: pad the audio to 30 s, then transform), the encoder, the
//!    greedy decode loop ([`decode_window`]) over the cached decoder
//!    ([`CachedDecoder`]), and [`align_words`] on the cross-attention of the
//!    alignment heads.
//!
//! ## Where things run
//!
//! CPU only. `ExecProvider::default_for_platform()` would pick CoreML on
//! macOS; the merged decoder's `If` node and its dynamic shapes are not
//! something CoreML is known to run, and this has been measured on Linux CPU
//! only (see the PR for #384), so the choice is `Cpu` until someone measures
//! the other.
//!
//! The two sessions live in a process-wide [`ModelRegistry`], keyed by the
//! file's canonical path, size and modification time, so loading the same
//! folder twice does not hold two copies of 136 MB of weights, and replacing
//! a file gets a new session.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ml_pipeline::{ExecProvider, ModelRegistry, SharedSession};

use crate::cached::CachedDecoder;
use crate::chunk::{run_windows, WindowResult, WindowTranscriber};
use crate::decode::{decode_window, DecodeConfig, StopReason, TokenIds, WindowDecode};
use crate::export::{self, alignment_heads, ExportLayout};
use crate::mel::{MelExtractor, N_SAMPLES};
use crate::ort_io::{run_encoder, specs, OrtDecoder};
use crate::refine::{is_silent, trim_to_sound};
use crate::timing::{
    align_words, encoder_frames, merge_punctuations, AlignmentInput, WordSplit, WordTiming,
    APPEND_PUNCTUATIONS, PREPEND_PUNCTUATIONS, SECONDS_PER_FRAME,
};
use crate::tokenizer::{Piece, Tokenizer};
use crate::{resample_to_16khz_mono, Result, WhisperError, Word};

/// Where each file of the export is, inside the model folder.
pub const ENCODER_FILE: &str = "onnx/encoder_model.onnx";
pub const DECODER_FILE: &str = "onnx/decoder_model_merged_int8.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// `[SOT, language, transcribe, notimestamps]`.
const PROMPT_LEN: usize = 4;

/// The words of a recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// In order, with times in seconds from the start of the audio.
    pub words: Vec<Word>,
    /// The language code (`"en"`), given or detected from the first window that
    /// had speech; `None` if no window had any.
    pub language: Option<String>,
    /// How many 30 s windows were decoded.
    pub windows: usize,
}

/// A loaded Whisper model: two ONNX sessions, the tokenizer and the settings
/// that go with this export.
///
/// `Send + Sync`: the sessions sit behind mutexes, so two threads transcribing
/// with one model take turns on each session.
pub struct WhisperModel {
    encoder: SharedSession,
    decoder: SharedSession,
    layout: ExportLayout,
    heads: &'static [(usize, usize)],
    tokenizer: Tokenizer,
    config: DecodeConfig,
    mel: MelExtractor,
}

// `ort::session::Session` doesn't implement `Debug`; provide a manual
// stub so callers can use `.unwrap()`/`.expect_err()` in tests.
impl std::fmt::Debug for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperModel")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

fn registry() -> &'static ModelRegistry {
    static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ModelRegistry::new)
}

/// A registry id that changes when the file does, and that names the role:
/// the same file given to both roles (a wrong folder, or symlinks) is two
/// sessions, not one session locked twice.
fn session_id(role: &str, path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let (len, modified) = std::fs::metadata(path)
        .map(|m| {
            let nanos = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            (m.len(), nanos)
        })
        .unwrap_or((0, 0));
    format!("whisper:{role}:{}:{len}:{modified}", canonical.display())
}

/// Turn the pipeline's error for loading `file` into ours.
fn load_error(e: ml_pipeline::Error, file: &Path) -> WhisperError {
    use ml_pipeline::Error as E;
    match e {
        E::MissingRuntime { .. } | E::RuntimeLoad { .. } | E::RuntimeUnsupported { .. } => {
            WhisperError::RuntimeUnavailable(e.to_string())
        }
        E::Io(io) if io.kind() == std::io::ErrorKind::NotFound => WhisperError::ModelMissing {
            path: file.display().to_string(),
        },
        other => WhisperError::ModelLoad(format!("{}: {other}", file.display())),
    }
}

impl WhisperModel {
    /// Open the model in `dir`, the folder of the pinned export.
    ///
    /// Returns [`WhisperError::ModelMissing`] if `dir` does not exist (so
    /// callers can present a structured hint instead of an opaque ORT error),
    /// [`WhisperError::RuntimeUnavailable`] if there is no ONNX Runtime library
    /// to load, [`WhisperError::UnexpectedExport`] if the folder is not the
    /// export this build reads (a file is missing, or a session's inputs and
    /// outputs differ), and [`WhisperError::ModelLoad`] for any other failure of
    /// the runtime to open a file.
    ///
    /// Nothing is downloaded: the folder is the user's, from Settings, or a
    /// developer's (`WHISPER_MODEL_PATH`).
    pub fn load(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            return Err(WhisperError::ModelMissing {
                path: dir.display().to_string(),
            });
        }
        // Before any `ort` call: `Session::builder()` panics, rather than
        // erroring, when the library is missing. `ModelRegistry::load` asks
        // too. Asking here, before the folder is looked at, makes the answer on
        // a machine without the runtime the same whatever the path holds
        // (`tests/runtime_missing.rs` and the tools' `ml_runtime_missing` point
        // it at a file that is not a model).
        ml_pipeline::runtime::ensure()
            .map_err(|e| WhisperError::RuntimeUnavailable(e.to_string()))?;

        let files = locate(dir)?;

        let tokenizer = Tokenizer::from_file(&files.tokenizer)?;
        let ids = token_ids(&tokenizer)?;

        let encoder = registry()
            .load(
                &session_id("encoder", &files.encoder),
                &files.encoder,
                ExecProvider::Cpu,
            )
            .map_err(|e| load_error(e, &files.encoder))?;
        let decoder = registry()
            .load(
                &session_id("decoder", &files.decoder),
                &files.decoder,
                ExecProvider::Cpu,
            )
            .map_err(|e| load_error(e, &files.decoder))?;

        // One lock at a time.
        let (enc_inputs, enc_outputs) = {
            let enc = encoder.lock().map_err(poisoned)?;
            (specs(enc.inputs()), specs(enc.outputs()))
        };
        let (dec_inputs, dec_outputs) = {
            let dec = decoder.lock().map_err(poisoned)?;
            (specs(dec.inputs()), specs(dec.outputs()))
        };
        export::check_encoder(&enc_inputs, &enc_outputs).map_err(WhisperError::UnexpectedExport)?;
        let frames_dims = enc_outputs
            .iter()
            .find(|o| o.name == "last_hidden_state")
            .map(|o| o.dims.clone())
            .unwrap_or_default();
        let layout = export::check_decoder(&dec_inputs, &dec_outputs, &frames_dims)
            .map_err(WhisperError::UnexpectedExport)?;
        if layout.vocab != ids.vocab_size || tokenizer.vocab_size() != layout.vocab {
            return Err(WhisperError::UnexpectedExport(format!(
                "the decoder has {} logits but tokenizer.json has {} tokens",
                layout.vocab,
                tokenizer.vocab_size()
            )));
        }
        let heads = alignment_heads(layout.layers, layout.heads)
            .expect("check_decoder refused a layout without known alignment heads");

        let mut config = DecodeConfig::multilingual();
        config.tokens = ids;
        // The no-speech decision is made here, after the window is decoded
        // (`skipped`), not in the loop, which would skip on the probability alone.
        config.no_speech_threshold = None;
        Ok(Self {
            encoder,
            decoder,
            layout,
            heads,
            tokenizer,
            config,
            mel: MelExtractor::new(),
        })
    }

    /// The decoder's geometry, as read from the files.
    pub fn layout(&self) -> &ExportLayout {
        &self.layout
    }

    /// The token for a language code (`"en"`), or an error if the model has
    /// no such language.
    fn language_token(&self, code: &str) -> Result<u32> {
        language_token(&self.tokenizer, &self.config.tokens.languages, code)
    }

    /// Run the Phase-1 stub.
    ///
    /// The `transcribe` tool still calls this, and answers "not implemented in
    /// this build" because of it. The working path is the free function
    /// [`transcribe`], which the tool does not call yet; wiring it in is the
    /// next change (#384), together with the tool schema, the download prompt
    /// and every line of copy that says transcription is unavailable.
    ///
    /// Input validation still runs first: a caller who forgot the resampler
    /// should hear about *that*, which is a mistake they can fix, rather than
    /// about the stub.
    pub fn transcribe(&self, audio_16khz_mono: &[f32]) -> Result<Vec<Word>> {
        // Sanity-check the input shape so downstream callers get a
        // clear error if they forget the resampler.
        if !audio_16khz_mono.iter().all(|s| s.is_finite()) {
            return Err(WhisperError::InvalidAudio(
                "audio buffer contains non-finite samples".into(),
            ));
        }
        // An empty `Ok` here would be a success carrying zero words, which a
        // caller cannot tell from "this recording has no speech" (#233).
        Err(WhisperError::NotImplemented)
    }
}

/// The three files of the export, found.
struct Files {
    encoder: PathBuf,
    decoder: PathBuf,
    tokenizer: PathBuf,
}

/// Find the export's files in `dir`, or say which is missing.
fn locate(dir: &Path) -> Result<Files> {
    let file = |name: &str| -> Result<PathBuf> {
        let path = dir.join(name);
        if path.is_file() {
            Ok(path)
        } else {
            Err(WhisperError::UnexpectedExport(format!(
                "{} has no {name}; the folder has to hold the files of \
                 onnx-community/whisper-base_timestamped ({ENCODER_FILE}, {DECODER_FILE} and \
                 {TOKENIZER_FILE})",
                dir.display()
            )))
        }
    };
    Ok(Files {
        encoder: file(ENCODER_FILE)?,
        decoder: file(DECODER_FILE)?,
        tokenizer: file(TOKENIZER_FILE)?,
    })
}

/// The token for a language code (`"en"`) among the language tokens
/// `languages`, or [`WhisperError::UnknownLanguage`].
fn language_token(tokenizer: &Tokenizer, languages: &Range<u32>, code: &str) -> Result<u32> {
    tokenizer
        .special_id(&format!("<|{}|>", code.trim().to_ascii_lowercase()))
        .filter(|t| languages.contains(t))
        .ok_or_else(|| WhisperError::UnknownLanguage(code.to_string()))
}

/// `"en"` for the token `<|en|>`.
fn language_code(tokenizer: &Tokenizer, token: u32) -> Option<String> {
    match tokenizer.pieces(&[token]).ok()?.pop()? {
        Piece::Control(name) => name
            .strip_prefix("<|")
            .and_then(|n| n.strip_suffix("|>"))
            .map(str::to_string),
        _ => None,
    }
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> WhisperError {
    WhisperError::Inference("a session's lock is poisoned".into())
}

/// The special-token ids of this tokenizer, which have to be the ones the
/// decode loop's defaults and the export's prompt use.
fn token_ids(tokenizer: &Tokenizer) -> Result<TokenIds> {
    let expected = TokenIds::multilingual();
    let named = |name: &str| tokenizer.special_id(name);
    let checks = [
        ("<|endoftext|>", expected.eot),
        ("<|startoftranscript|>", expected.sot),
        ("<|transcribe|>", expected.transcribe),
        ("<|nocaptions|>", expected.no_speech),
        ("<|notimestamps|>", expected.no_timestamps),
        ("<|0.00|>", expected.timestamp_begin),
        ("<|en|>", expected.languages.start),
    ];
    for (name, id) in checks {
        if named(name) != Some(id) {
            return Err(WhisperError::UnexpectedExport(format!(
                "tokenizer.json numbers {name} as {:?}, this build expects {id}: it is not the \
                 multilingual tokenizer of the pinned export",
                named(name)
            )));
        }
    }
    if tokenizer.vocab_size() != expected.vocab_size {
        return Err(WhisperError::UnexpectedExport(format!(
            "tokenizer.json has {} tokens, this build expects {}",
            tokenizer.vocab_size(),
            expected.vocab_size
        )));
    }
    Ok(expected)
}

/// A window is passed over as having no speech when the model's probability of
/// `<|nocaptions|>` at the start is above this...
pub(crate) const NO_SPEECH_THRESHOLD: f32 = 0.6;
/// ...and what it nevertheless wrote is not confident: the mean log-probability
/// of its tokens is at most this.
pub(crate) const LOGPROB_THRESHOLD: f32 = -1.0;

/// The mean natural-log probability of the text tokens of a decoded window.
/// `whisper.transcribe`'s `avg_logprob` also counts `<|endoftext|>`, which the
/// decode loop does not keep; leaving it out makes this the lower of the two,
/// so a window is skipped a little more readily than there.
fn average_logprob(decoded: &WindowDecode) -> f32 {
    if decoded.token_probs.is_empty() {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = decoded
        .token_probs
        .iter()
        .map(|&p| f64::from(p).max(f64::MIN_POSITIVE).ln())
        .sum();
    (sum / decoded.token_probs.len() as f64) as f32
}

/// Whether a decoded window is passed over as having no speech.
///
/// The plan on #384 skipped on the probability alone (above 0.6 at the start
/// of the window). On the real model that threw away windows it transcribes
/// correctly: a 30 s window of clear synthetic speech had `P(nospeech)` of
/// 0.86 to 0.99 though the same voice in a 7 s clip had 0.11 (see the PR),
/// and the whole recording came back empty. `whisper.transcribe` keeps the
/// window when its text is confident, whatever the probability, and so does
/// this: **both** have to hold. Noise at any level, which the model writes
/// "1.5%" or "you" for at mean log-probabilities of -1.5 to -2, is still passed
/// over; the numbers are in the PR.
fn skipped(decoded: &WindowDecode) -> bool {
    decoded.no_speech_prob > NO_SPEECH_THRESHOLD && average_logprob(decoded) <= LOGPROB_THRESHOLD
}

/// The windows of one run: the model and the cancel flag.
struct ModelWindows<'a> {
    model: &'a WhisperModel,
    cancel: &'a dyn Fn() -> bool,
}

impl WindowTranscriber for ModelWindows<'_> {
    fn window(&mut self, audio: &[f32], language: Option<u32>) -> Result<WindowResult> {
        let m = self.model;
        if is_silent(audio) {
            // Nothing to hear, so nothing to ask the model, and no language to learn.
            return Ok(WindowResult {
                words: Vec::new(),
                language: language.unwrap_or(m.config.tokens.languages.start),
                speech: false,
                finished: true,
            });
        }
        let mel = m.mel.log_mel_window(audio).into_vec();
        let encoder = run_encoder(&m.encoder, mel, &m.layout)?;

        let run = OrtDecoder::lock(&m.decoder, m.layout)?;
        let mut decoder = CachedDecoder::new(run, m.layout, m.heads, self.cancel);
        let decoded = decode_window(&mut decoder, &encoder, &m.config, language)
            .map_err(WhisperError::from)?;

        if skipped(&decoded) || decoded.tokens.is_empty() {
            return Ok(WindowResult {
                words: Vec::new(),
                language: decoded.language,
                speech: false,
                finished: true,
            });
        }

        decoder.complete_for(&decoded.tokens, PROMPT_LEN, &encoder)?;
        let content_frames = encoder_frames(audio.len().min(N_SAMPLES)).clamp(1, m.layout.frames);
        let heads = decoder.attention(
            PROMPT_LEN - 1,
            PROMPT_LEN + decoded.tokens.len(),
            content_frames,
        )?;

        let code = language_code(&m.tokenizer, decoded.language).unwrap_or_default();
        let split = WordSplit::for_language(&code);
        let timings = align_words(
            &AlignmentInput {
                heads: &heads,
                content_frames,
                tokens: &decoded.tokens,
                token_probs: &decoded.token_probs,
            },
            |t| m.tokenizer.token_bytes(t),
            split,
        )?;
        let timings = merge_punctuations(timings, PREPEND_PUNCTUATIONS, APPEND_PUNCTUATIONS);
        let window_s = audio.len() as f32 / crate::mel::SAMPLE_RATE as f32;
        let mut words: Vec<Word> = timings
            .into_iter()
            .filter_map(|t| word_from(t, window_s))
            .collect();
        trim_to_sound(&mut words, audio);
        Ok(WindowResult {
            words,
            language: decoded.language,
            speech: true,
            finished: decoded.stop == StopReason::EndOfText,
        })
    }
}

/// A [`Word`] from an aligned one: the leading space is trimmed, a word with
/// no text is dropped, and a word is at least one encoder frame long and ends
/// inside the window.
fn word_from(t: WordTiming, window_s: f32) -> Option<Word> {
    let text = t.text.trim();
    if text.is_empty() {
        return None;
    }
    let start_s = t.start_s.clamp(0.0, window_s);
    let end_s = t
        .end_s
        .max(start_s + SECONDS_PER_FRAME)
        .min(window_s.max(start_s + SECONDS_PER_FRAME));
    Some(Word {
        text: text.to_string(),
        start_s,
        end_s,
        confidence: t.probability.clamp(0.0, 1.0),
    })
}

/// Transcribe `samples`, interleaved at `sample_rate` Hz with `channels`
/// channels, into words with times.
///
/// `language` is a language code (`"en"`), or `None` to detect it from the
/// first window that has speech. `cancel` is asked before every window and
/// before every decoder step; when it answers `true` the call ends with
/// [`WhisperError::Cancelled`] and returns nothing.
///
/// Slow work: a 30 s window costs one encoder pass and up to 224 decoder
/// steps. Call it off any lock the app holds.
///
/// ## What the words are
///
/// `text` has its leading space trimmed and its punctuation attached
/// (`"Hello,"`). `start_s`/`end_s` are seconds from the start of `samples`,
/// estimated from the decoder's cross-attention (a DTW over a few attention
/// heads, as `whisper.timing.find_alignment` does), not measured: a start is
/// typically within a tenth of a second, and an end is where the next word
/// starts, so a word followed by a pause is stretched over the pause.
/// `confidence` is the mean probability the model gave the word's tokens.
pub fn transcribe(
    model: &WhisperModel,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    language: Option<&str>,
    cancel: impl Fn() -> bool,
) -> Result<Transcript> {
    if !samples.iter().all(|s| s.is_finite()) {
        return Err(WhisperError::InvalidAudio(
            "audio buffer contains non-finite samples".into(),
        ));
    }
    let language = language
        .map(|code| model.language_token(code))
        .transpose()?;
    let mono = resample_to_16khz_mono(samples, sample_rate, channels)?;
    let mut windows = ModelWindows {
        model,
        cancel: &cancel,
    };
    let chunked = run_windows(&mut windows, &mono, language, &cancel)?;
    Ok(Transcript {
        words: chunked.words,
        language: chunked
            .language
            .and_then(|t| language_code(&model.tokenizer, t)),
        windows: chunked.windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        Tokenizer::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer_tiny.json"),
        )
        .expect("the tiny tokenizer fixture loads")
    }

    fn timing(text: &str, start_s: f32, end_s: f32) -> WordTiming {
        WordTiming {
            text: text.to_string(),
            tokens: 0..1,
            start_s,
            end_s,
            probability: 0.8,
        }
    }

    fn decoded(no_speech_prob: f32, probs: &[f32]) -> WindowDecode {
        WindowDecode {
            language: 50_259,
            language_prob: None,
            tokens: (0..probs.len() as u32).collect(),
            token_probs: probs.to_vec(),
            no_speech_prob,
            stop: StopReason::EndOfText,
        }
    }

    #[test]
    fn a_window_is_skipped_only_if_it_sounds_empty_and_reads_unsure() {
        let sure = [0.9, 0.8, 0.95]; // mean ln p about -0.13
        let unsure = [0.2, 0.25]; // mean ln p about -1.5
                                  // Both: skipped. This is what noise gets.
        assert!(skipped(&decoded(0.95, &unsure)));
        // A model that says "no speech" but writes a confident transcript is
        // believed (the 30 s word list: 0.95 and a clean transcript).
        assert!(!skipped(&decoded(0.95, &sure)));
        // A model that does not say it, however unsure it is: kept. (A tone
        // got "." at 0.598 and -4.5; the threshold cannot catch that.)
        assert!(!skipped(&decoded(0.3, &unsure)));
        // The boundaries: the probability must be above, the mean at or below.
        assert!(!skipped(&decoded(NO_SPEECH_THRESHOLD, &unsure)));
        let at = [(LOGPROB_THRESHOLD).exp()];
        assert!(skipped(&decoded(0.7, &at)));
        let above = [(LOGPROB_THRESHOLD + 0.01).exp()];
        assert!(!skipped(&decoded(0.7, &above)));
        // No text at all has nothing to be confident about.
        assert!(skipped(&decoded(0.7, &[])));
        assert!(!skipped(&decoded(0.2, &[])));
    }

    #[test]
    fn the_average_is_the_mean_of_the_natural_logs() {
        let d = decoded(0.0, &[0.5, 0.25]);
        let want = (0.5f32.ln() + 0.25f32.ln()) / 2.0;
        assert!((average_logprob(&d) - want).abs() < 1e-6);
        // A probability of exactly zero does not make it -inf-NaN.
        assert!(average_logprob(&decoded(0.0, &[0.0, 0.5])).is_finite());
    }

    #[test]
    fn the_same_file_in_two_roles_is_two_sessions() {
        // Given one file for both, the registry used to return one session for
        // both and `load` locked it twice. (A symlink resolves to the same
        // canonical path.)
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("m.onnx");
        std::fs::write(&file, b"x").unwrap();
        assert_ne!(session_id("encoder", &file), session_id("decoder", &file));
        assert_eq!(session_id("encoder", &file), session_id("encoder", &file));
        // And a changed file is a new session.
        let before = session_id("encoder", &file);
        std::fs::write(&file, b"xy").unwrap();
        assert_ne!(before, session_id("encoder", &file));
    }

    #[test]
    fn a_folder_with_a_file_missing_says_which() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
        std::fs::write(dir.path().join(ENCODER_FILE), b"x").unwrap();
        std::fs::write(dir.path().join(TOKENIZER_FILE), b"{}").unwrap();
        let err = locate(dir.path()).err().expect("the decoder is missing");
        let WhisperError::UnexpectedExport(why) = &err else {
            panic!("{err:?}");
        };
        assert!(why.contains(DECODER_FILE), "{why}");
        assert!(why.contains("whisper-base_timestamped"), "{why}");

        std::fs::write(dir.path().join(DECODER_FILE), b"x").unwrap();
        assert!(locate(dir.path()).is_ok());
    }

    #[test]
    fn a_path_that_is_a_file_is_not_a_model_folder() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.onnx");
        std::fs::write(&file, b"x").unwrap();
        assert!(matches!(
            locate(&file).err(),
            Some(WhisperError::UnexpectedExport(_))
        ));
    }

    #[test]
    fn a_tokenizer_numbered_differently_is_refused_by_name() {
        // The tiny fixture numbers <|endoftext|> as 11, not 50257.
        let err = token_ids(&tiny_tokenizer()).unwrap_err();
        let WhisperError::UnexpectedExport(why) = &err else {
            panic!("{err:?}");
        };
        assert!(
            why.contains("<|endoftext|>") && why.contains("50257"),
            "{why}"
        );
    }

    #[test]
    fn language_codes_map_to_tokens_and_back() {
        let tokenizer = tiny_tokenizer();
        let languages = 13..15; // <|en|>, <|de|>
        assert_eq!(language_token(&tokenizer, &languages, "en").unwrap(), 13);
        assert_eq!(language_token(&tokenizer, &languages, " DE ").unwrap(), 14);
        for bad in ["xx", "", "transcribe", "0.00"] {
            let err = language_token(&tokenizer, &languages, bad).unwrap_err();
            assert!(
                matches!(err, WhisperError::UnknownLanguage(_)),
                "{bad}: {err:?}"
            );
        }
        assert_eq!(language_code(&tokenizer, 13).as_deref(), Some("en"));
        assert_eq!(language_code(&tokenizer, 14).as_deref(), Some("de"));
        assert_eq!(
            language_code(&tokenizer, 0),
            None,
            "a text token is no language"
        );
    }

    #[test]
    fn a_word_keeps_its_text_trimmed_and_a_positive_length_inside_the_window() {
        let w = word_from(timing(" Hello,", 1.0, 1.5), 30.0).unwrap();
        assert_eq!(w.text, "Hello,");
        assert_eq!((w.start_s, w.end_s), (1.0, 1.5));
        assert!((w.confidence - 0.8).abs() < 1e-6);

        // Two tokens that start on the same frame: the first has no length.
        let w = word_from(timing(" a", 2.0, 2.0), 30.0).unwrap();
        assert!(w.end_s > w.start_s, "{w:?}");
        assert!((w.end_s - 2.02).abs() < 1e-6);

        // Past the window's end is pulled back, to at least a frame.
        let w = word_from(timing(" z", 29.99, 31.0), 30.0).unwrap();
        assert!(w.start_s < w.end_s && w.end_s <= 30.01, "{w:?}");

        // A word of only spaces is dropped.
        assert!(word_from(timing("  ", 1.0, 2.0), 30.0).is_none());
    }
}
