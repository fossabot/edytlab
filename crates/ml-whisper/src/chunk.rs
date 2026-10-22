//! Long audio, one 30 s window at a time (plan step 5 on #384).
//!
//! The encoder takes exactly 30 s. A recording longer than that is cut into
//! windows, and the rule for where the next one starts is the whole of this
//! module:
//!
//! * a window is up to 30 s of audio from `seek`;
//! * if it is the **last** one (it reaches the end of the audio), every word
//!   is kept;
//! * otherwise the last word may have been cut by the window's edge, and the
//!   model's idea of it is a guess. A word counts as cut when the decoder did
//!   not finish by itself (it ran out of tokens, or looped and was stopped), or
//!   when the word ends within [`EDGE_S`] of the window's end. A cut word is
//!   dropped, and the next window starts where the **last complete word
//!   ended**, so the word is heard whole the second time;
//! * with no complete word at all, the next window starts at the cut word's
//!   own start if that is more than [`MIN_ADVANCE_S`] in, and a whole window
//!   on otherwise (so a silent or hopeless window costs 30 s and never loops);
//! * the next window never starts less than [`MIN_ADVANCE_S`] after this one;
//! * a window that decoded to nothing, or was skipped as no speech, is
//!   passed over whole.
//!
//! Word times in a window are relative to its start; they are offset by
//! `seek` here. The language is fixed by the first window that has speech and
//! then given to every later one, as `whisper.transcribe` does, so a window of
//! music cannot flip the language of the rest.
//!
//! The windows themselves are decoded by a [`WindowTranscriber`], which the
//! model implements and the tests fake: the rule above is checked on 75 s of
//! made-up audio in CI.

use crate::mel::{N_SAMPLES, SAMPLE_RATE};
use crate::{Result, WhisperError, Word};

/// Samples in a full window: 30 s at 16 kHz.
pub(crate) const WINDOW_SAMPLES: usize = N_SAMPLES;
/// A window shorter than this (0.1 s) is not worth an encoder pass.
const MIN_WINDOW_SAMPLES: usize = SAMPLE_RATE / 10;
/// A word that ends within this many seconds of a full window's end may have
/// been cut by it.
pub(crate) const EDGE_S: f32 = 0.5;
/// The next window starts at least this many seconds after the last.
pub(crate) const MIN_ADVANCE_S: f32 = 0.5;

/// What decoding one window gave.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowResult {
    /// Times in seconds from the window's start.
    pub words: Vec<Word>,
    /// The language token used.
    pub language: u32,
    /// Whether the window had speech: it decoded at least one token. Only such
    /// a window sets the language.
    pub speech: bool,
    /// The decoder ended the window itself, at `<|endoftext|>`.
    pub finished: bool,
}

/// Decodes one window of 16 kHz mono audio, up to 30 s.
pub(crate) trait WindowTranscriber {
    /// `language` is the language token to use, or `None` to detect it.
    fn window(&mut self, audio: &[f32], language: Option<u32>) -> Result<WindowResult>;
}

/// The words of a whole recording.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Chunked {
    pub words: Vec<Word>,
    /// The language token, if any window had speech (or one was given).
    pub language: Option<u32>,
    /// Windows decoded.
    pub windows: usize,
}

/// Split a full window's words into the ones to keep and where the next
/// window starts (seconds from this window's start; `None` means the end of
/// this window).
fn cut_window(mut words: Vec<Word>, window_s: f32, finished: bool) -> (Vec<Word>, Option<f32>) {
    let Some(last) = words.last() else {
        return (words, None);
    };
    let cut = !finished || last.end_s > window_s - EDGE_S;
    if !cut {
        let next = last.end_s;
        return (words, Some(next));
    }
    let cut_word = words.pop().expect("there is a last word");
    match words.last() {
        Some(complete) => {
            let next = complete.end_s;
            (words, Some(next))
        }
        None if cut_word.start_s > MIN_ADVANCE_S => (words, Some(cut_word.start_s)),
        None => (words, None),
    }
}

/// Transcribe `audio` (16 kHz mono) window by window.
///
/// `cancel` is asked before each window.
pub(crate) fn run_windows<T: WindowTranscriber>(
    transcriber: &mut T,
    audio: &[f32],
    mut language: Option<u32>,
    cancel: &dyn Fn() -> bool,
) -> Result<Chunked> {
    let total = audio.len();
    let rate = SAMPLE_RATE as f32;
    let mut words = Vec::new();
    let mut windows = 0;
    let mut seek = 0usize;
    while total - seek >= MIN_WINDOW_SAMPLES {
        if cancel() {
            return Err(WhisperError::Cancelled);
        }
        let end = (seek + WINDOW_SAMPLES).min(total);
        let slice = &audio[seek..end];
        let result = transcriber.window(slice, language)?;
        windows += 1;
        if result.speech && language.is_none() {
            language = Some(result.language);
        }

        let last_window = end == total;
        let window_s = slice.len() as f32 / rate;
        let (keep, next) = if last_window {
            (result.words, None)
        } else {
            cut_window(result.words, window_s, result.finished)
        };
        let offset = seek as f32 / rate;
        words.extend(keep.into_iter().map(|w| Word {
            start_s: w.start_s + offset,
            end_s: w.end_s + offset,
            ..w
        }));
        if last_window {
            break;
        }
        seek = match next {
            Some(s) => {
                let advance = (s.max(MIN_ADVANCE_S) * rate).round() as usize;
                // A word's end can lie past the window's if the aligner
                // overshoots; the window's end is as far as it can go.
                seek + advance.min(end - seek)
            }
            None => end,
        };
    }
    Ok(Chunked {
        words,
        language,
        windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const RATE: usize = SAMPLE_RATE;

    /// Audio whose sample `i` is `i`, so a window can tell where it starts.
    fn ramp(seconds: f32) -> Vec<f32> {
        (0..(seconds * RATE as f32) as usize)
            .map(|i| i as f32)
            .collect()
    }

    /// Word `k` is spoken from `2k + 0.1` s to `2k + 1.4` s.
    fn word_span(k: usize) -> (f32, f32) {
        (2.0 * k as f32 + 0.1, 2.0 * k as f32 + 1.4)
    }

    /// A fake model that hears the words above, the way Whisper hears them: a
    /// word that starts in the window is in it, and one that runs past the end
    /// is heard as ending there.
    struct Hears {
        /// Seconds before which nothing is heard (music, silence).
        speech_from: f32,
        /// `(window start in seconds, language passed in)` for each call.
        calls: RefCell<Vec<(f32, Option<u32>)>>,
        finished: bool,
    }

    impl Hears {
        fn new() -> Self {
            Self {
                speech_from: 0.0,
                calls: RefCell::new(Vec::new()),
                finished: true,
            }
        }
    }

    impl WindowTranscriber for Hears {
        fn window(&mut self, audio: &[f32], language: Option<u32>) -> Result<WindowResult> {
            let start = audio[0] / RATE as f32;
            let len = audio.len() as f32 / RATE as f32;
            self.calls.borrow_mut().push((start, language));
            let mut words = Vec::new();
            for k in 0.. {
                let (s, e) = word_span(k);
                if s >= start + len {
                    break;
                }
                if s < start.max(self.speech_from) {
                    continue;
                }
                words.push(Word {
                    text: format!("w{k}"),
                    start_s: s - start,
                    end_s: (e - start).min(len),
                    confidence: 0.9,
                });
            }
            Ok(WindowResult {
                speech: !words.is_empty(),
                words,
                language: 7,
                finished: self.finished,
            })
        }
    }

    fn texts(words: &[Word]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    fn never() -> bool {
        false
    }

    #[test]
    fn seventy_five_seconds_are_three_windows_and_every_word_is_heard_once() {
        let mut model = Hears::new();
        let out = run_windows(&mut model, &ramp(75.0), None, &never).unwrap();

        // Words 0..=37 start before 75 s (word 37 at 74.1 s).
        let expected: Vec<String> = (0..=37).map(|k| format!("w{k}")).collect();
        assert_eq!(texts(&out.words), expected, "each word once, in order");
        for (k, w) in out.words.iter().enumerate() {
            let (s, e) = word_span(k);
            assert!(
                (w.start_s - s).abs() < 1e-3,
                "{}: start {} vs {s}",
                w.text,
                w.start_s
            );
            // The last word is cut by the end of the audio.
            let e = if k == 37 { 75.0 } else { e };
            assert!(
                (w.end_s - e).abs() < 1e-3,
                "{}: end {} vs {e}",
                w.text,
                w.end_s
            );
        }

        // Window 1 [0, 30): word 14 ends at 29.4, 0.6 s from the edge, so it is
        // complete and the next window starts where it ended. Window 2
        // [29.4, 59.4): word 29 ends exactly at the edge, so it is cut, and the
        // next window starts at the end of word 28 (57.4). Window 3 reaches the
        // end of the audio.
        let starts: Vec<f32> = model.calls.borrow().iter().map(|c| c.0).collect();
        assert_eq!(starts.len(), 3, "{starts:?}");
        assert_eq!(out.windows, 3);
        for (got, want) in starts.iter().zip([0.0, 29.4, 57.4]) {
            assert!((got - want).abs() < 1e-3, "{starts:?}");
        }
    }

    #[test]
    fn a_word_cut_by_the_window_is_heard_whole_in_the_next() {
        struct LateWord(Vec<f32>);
        impl WindowTranscriber for LateWord {
            fn window(&mut self, audio: &[f32], _: Option<u32>) -> Result<WindowResult> {
                let start = audio[0] / RATE as f32;
                self.0.push(start);
                // One word starts at 27 s and ends at 33 s: cut by the first
                // window, whole in the second.
                let (s, e) = (27.0f32, 33.0f32);
                let len = audio.len() as f32 / RATE as f32;
                let mut words = vec![Word {
                    text: "early".into(),
                    start_s: 10.0 - start,
                    end_s: 11.0 - start,
                    confidence: 1.0,
                }];
                if start > 10.0 {
                    words.clear();
                }
                if s >= start {
                    words.push(Word {
                        text: "long".into(),
                        start_s: s - start,
                        end_s: (e - start).min(len),
                        confidence: 1.0,
                    });
                }
                Ok(WindowResult {
                    words,
                    language: 1,
                    speech: true,
                    finished: true,
                })
            }
        }
        let mut model = LateWord(Vec::new());
        let out = run_windows(&mut model, &ramp(45.0), None, &never).unwrap();
        // First window [0, 30): "long" runs to the edge, so it is dropped and
        // the next window starts where "early" ended (11 s).
        assert!(model.0.len() >= 2, "{:?}", model.0);
        assert!((model.0[1] - 11.0).abs() < 1e-3, "{:?}", model.0);
        assert_eq!(texts(&out.words), vec!["early", "long"]);
        assert!((out.words[1].start_s - 27.0).abs() < 1e-3);
        assert!((out.words[1].end_s - 33.0).abs() < 1e-3);
    }

    #[test]
    fn a_decoder_that_did_not_finish_loses_its_last_word_to_the_next_window() {
        let mut model = Hears::new();
        model.finished = false;
        let out = run_windows(&mut model, &ramp(45.0), None, &never).unwrap();
        // Every word is still heard exactly once: the last of each window is
        // dropped and heard again.
        let n = out.words.len();
        let expected: Vec<String> = (0..n).map(|k| format!("w{k}")).collect();
        assert_eq!(texts(&out.words), expected);
        assert!(model.calls.borrow().len() >= 2);
    }

    #[test]
    fn windows_without_speech_are_passed_over_whole_and_leave_the_language_open() {
        let mut model = Hears::new();
        model.speech_from = 70.0; // music for the first 70 s
        let out = run_windows(&mut model, &ramp(75.0), None, &never).unwrap();
        let calls = model.calls.borrow().clone();
        assert_eq!(
            calls.iter().map(|c| c.0).collect::<Vec<_>>(),
            vec![0.0, 30.0, 60.0],
            "a silent window costs exactly 30 s"
        );
        // The language is still unknown until a window has speech, and is
        // then the one that window found.
        assert_eq!(
            calls.iter().map(|c| c.1).collect::<Vec<_>>(),
            vec![None, None, None]
        );
        assert_eq!(out.language, Some(7));
        assert_eq!(texts(&out.words), vec!["w35", "w36", "w37"]);
    }

    #[test]
    fn the_language_of_the_first_window_with_speech_is_given_to_the_rest() {
        let mut model = Hears::new();
        run_windows(&mut model, &ramp(75.0), None, &never).unwrap();
        let langs: Vec<_> = model.calls.borrow().iter().map(|c| c.1).collect();
        assert_eq!(langs, vec![None, Some(7), Some(7)]);

        let mut model = Hears::new();
        run_windows(&mut model, &ramp(75.0), Some(3), &never).unwrap();
        let langs: Vec<_> = model.calls.borrow().iter().map(|c| c.1).collect();
        assert_eq!(
            langs,
            vec![Some(3), Some(3), Some(3)],
            "a given language is kept"
        );
    }

    #[test]
    fn a_short_recording_is_one_window_and_a_sliver_is_none() {
        let mut model = Hears::new();
        let out = run_windows(&mut model, &ramp(5.0), None, &never).unwrap();
        assert_eq!(out.windows, 1);
        assert_eq!(texts(&out.words), vec!["w0", "w1", "w2"]);

        let mut model = Hears::new();
        let out = run_windows(&mut model, &ramp(0.05), None, &never).unwrap();
        assert_eq!(out.windows, 0, "50 ms is not worth an encoder pass");
        assert!(out.words.is_empty());

        let out = run_windows(&mut model, &[], None, &never).unwrap();
        assert_eq!(out.windows, 0);
    }

    #[test]
    fn cancel_is_asked_before_every_window_and_stops_the_run() {
        let asked = Cell::new(0);
        let cancel = || {
            asked.set(asked.get() + 1);
            asked.get() > 1 // allow the first window, refuse the second
        };
        let mut model = Hears::new();
        let err = run_windows(&mut model, &ramp(75.0), None, &cancel).unwrap_err();
        assert!(matches!(err, WhisperError::Cancelled), "{err:?}");
        assert_eq!(model.calls.borrow().len(), 1);
    }

    #[test]
    fn a_window_that_never_gets_anywhere_still_ends() {
        // A model that hears one word ending at 0.1 s and gives up each time
        // (not finished): the next window must still start later, and the run
        // must end.
        struct Stuck(usize);
        impl WindowTranscriber for Stuck {
            fn window(&mut self, _: &[f32], _: Option<u32>) -> Result<WindowResult> {
                self.0 += 1;
                Ok(WindowResult {
                    words: vec![
                        Word {
                            text: "a".into(),
                            start_s: 0.0,
                            end_s: 0.1,
                            confidence: 1.0,
                        },
                        Word {
                            text: "b".into(),
                            start_s: 0.1,
                            end_s: 0.2,
                            confidence: 1.0,
                        },
                    ],
                    language: 1,
                    speech: true,
                    finished: false,
                })
            }
        }
        let mut model = Stuck(0);
        let out = run_windows(&mut model, &ramp(60.0), None, &never).unwrap();
        // 0.5 s is the least a window advances.
        assert!(model.0 <= 120 + 1, "{} windows", model.0);
        assert_eq!(out.windows, model.0);
    }

    #[test]
    fn a_lone_cut_word_late_in_the_window_is_heard_from_its_start() {
        // Silence, then a word that starts at 29.0 s and runs past the edge.
        struct Late;
        impl WindowTranscriber for Late {
            fn window(&mut self, audio: &[f32], _: Option<u32>) -> Result<WindowResult> {
                let start = audio[0] / RATE as f32;
                let len = audio.len() as f32 / RATE as f32;
                let words = if start < 29.0 {
                    vec![Word {
                        text: "late".into(),
                        start_s: 29.0 - start,
                        end_s: len,
                        confidence: 1.0,
                    }]
                } else {
                    vec![Word {
                        text: "late".into(),
                        start_s: 0.0,
                        end_s: 1.2,
                        confidence: 1.0,
                    }]
                };
                Ok(WindowResult {
                    words,
                    language: 1,
                    speech: true,
                    finished: true,
                })
            }
        }
        let out = run_windows(&mut Late, &ramp(40.0), None, &never).unwrap();
        assert_eq!(texts(&out.words), vec!["late"]);
        assert!(
            (out.words[0].start_s - 29.0).abs() < 1e-3,
            "{:?}",
            out.words
        );
        assert_eq!(out.windows, 2);
    }

    #[test]
    fn cutting_rules_on_their_own() {
        let w = |t: &str, s: f32, e: f32| Word {
            text: t.into(),
            start_s: s,
            end_s: e,
            confidence: 1.0,
        };
        // No words: pass over the whole window.
        assert_eq!(cut_window(Vec::new(), 30.0, true), (Vec::new(), None));
        // Last word well inside: keep all, resume at its end.
        let (keep, next) = cut_window(vec![w("a", 1.0, 2.0), w("b", 3.0, 20.0)], 30.0, true);
        assert_eq!((keep.len(), next), (2, Some(20.0)));
        // Last word at the edge: drop it, resume at the previous end.
        let (keep, next) = cut_window(vec![w("a", 1.0, 2.0), w("b", 3.0, 29.9)], 30.0, true);
        assert_eq!((texts(&keep), next), (vec!["a"], Some(2.0)));
        // Decoder gave up: the last word goes even if it ends early.
        let (keep, next) = cut_window(vec![w("a", 1.0, 2.0), w("b", 3.0, 10.0)], 30.0, false);
        assert_eq!((texts(&keep), next), (vec!["a"], Some(2.0)));
        // Only a cut word, started early: nothing is learned, move on.
        let (keep, next) = cut_window(vec![w("a", 0.2, 30.0)], 30.0, true);
        assert_eq!((keep.len(), next), (0, None));
    }
}
