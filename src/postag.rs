//! Cepstral's part-of-speech tagger, lifted from `ceplang_en.dll`.
//!
//! The tag decides a homograph's pronunciation: `record/vbp` is
//! `r ih0 k ao1 r d` and `record/nn` is `r eh1 k er0 d`. `CepLex::lookup_tag`
//! is the half that consumes a tag; this is the half that produces one.
//!
//! Two pieces, both extracted by `_out/pack_pos.py`:
//!
//! **The model** is a maximum-entropy classifier, 750,488 weighted
//! (feature, outcome) pairs over 492,535 features and the 45 Penn tags. No
//! feature names a previous tag, so there is no sequence search at all: each
//! token is an independent argmax of the summed weights of its active features,
//! and a pair the model does not carry contributes nothing.
//!
//! **The gate** is a list of 237 homographs. The engine decides per utterance,
//! and an utterance holding none of them is never tagged -- every `lex_lookup`
//! in it gets a NULL pos and the lexicon's `'0'` default. That is not an
//! optimisation to skip: it is why a 47-second passage of ordinary prose is
//! bit-exact without a tagger, and tagging those sentences anyway would change
//! output the engine never touched.
//!
//! Tokens reach the tagger lowercased, which is why `CTN_UPP` and `ALL_UPP` are
//! in the model but can never fire here -- they were trained on cased WSJ text
//! and the runtime never shows them a capital.

use std::io;
use std::path::Path;

const MAGIC: &[u8; 8] = b"CEPSPOS1";
/// One `(u8 outcome, f32 weight)` pair.
const REC: usize = 5;

pub struct PosTag {
    buf: Vec<u8>,
    outcomes: Vec<String>,
    n_feats: usize,
    hashes_off: usize,
    index_off: usize,
    recdata_off: usize,
    homographs: Vec<String>,
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64le(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

fn f32le(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn strtab(b: &[u8], o: &mut usize, n: usize) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = *b.get(*o)? as usize;
        *o += 1;
        out.push(String::from_utf8_lossy(b.get(*o..*o + len)?).into_owned());
        *o += len;
    }
    Some(out)
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Lowercase, with every digit replaced by `#`. The model's `NW0` template.
fn normalise(w: &str) -> String {
    w.chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .flat_map(|c| c.to_lowercase())
        .collect()
}

impl PosTag {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<PosTag> {
        let buf = std::fs::read(path)?;
        PosTag::from_bytes(buf)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad pos pack"))
    }

    pub fn from_bytes(buf: Vec<u8>) -> Option<PosTag> {
        if buf.len() < 8 + 9 * 4 || &buf[..8] != MAGIC {
            return None;
        }
        let g = |i: usize| u32le(&buf, 8 + i * 4) as usize;
        let (n_out, n_feats, n_hom) = (g(0), g(1), g(3));
        let (outcomes_off, hashes_off, index_off, recdata_off, hom_off) =
            (g(4), g(5), g(6), g(7), g(8));
        let mut o = outcomes_off;
        let outcomes = strtab(&buf, &mut o, n_out)?;
        let mut o = hom_off;
        let homographs = strtab(&buf, &mut o, n_hom)?;
        if hashes_off + 8 * n_feats > buf.len() || o > buf.len() {
            return None;
        }
        Some(PosTag { buf, outcomes, n_feats, hashes_off, index_off, recdata_off, homographs })
    }

    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    pub fn feature_count(&self) -> usize {
        self.n_feats
    }

    pub fn homographs(&self) -> &[String] {
        &self.homographs
    }

    /// Does this utterance reach the tagger at all?
    pub fn triggers<S: AsRef<str>>(&self, words: &[S]) -> bool {
        words.iter().any(|w| self.is_homograph(w.as_ref()))
    }

    pub fn is_homograph(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.homographs.binary_search_by(|p| p.as_str().cmp(lower.as_str())).is_ok()
    }

    /// The `(outcome, weight)` pairs the model carries for one feature.
    fn weights(&self, feat: &str) -> &[u8] {
        let want = fnv1a(feat);
        let (mut lo, mut hi) = (0usize, self.n_feats);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let h = u64le(&self.buf, self.hashes_off + mid * 8);
            if h == want {
                let a = u32le(&self.buf, self.index_off + mid * 4) as usize;
                let b = u32le(&self.buf, self.index_off + (mid + 1) * 4) as usize;
                return &self.buf[self.recdata_off + a..self.recdata_off + b];
            }
            if h < want {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        &[]
    }

    /// What the model carries for one feature, for tooling and tests.
    pub fn feature_weights(&self, feat: &str) -> Vec<(&str, f32)> {
        self.weights(feat)
            .chunks_exact(REC)
            .map(|r| (self.outcomes[r[0] as usize].as_str(), f32le(r, 1)))
            .collect()
    }

    /// Every feature the engine would build for `toks[i]`, all tokens already
    /// lowercased. Out of range positions are `BOS` and `EOS`.
    fn features(&self, toks: &[String], i: usize, out: &mut Vec<String>) {
        out.clear();
        let w = &toks[i];
        let at = |k: isize| -> &str {
            if k < 0 {
                "BOS"
            } else if k as usize >= toks.len() {
                "EOS"
            } else {
                &toks[k as usize]
            }
        };
        let (p1, p2) = (at(i as isize - 1), at(i as isize - 2));
        let (n1, n2) = (at(i as isize + 1), at(i as isize + 2));
        out.push(format!("W0_{w}"));
        out.push(format!("NW0_{}", normalise(w)));
        out.push(format!("W-1_{p1}"));
        out.push(format!("W-2_{p2}"));
        out.push(format!("W+1_{n1}"));
        out.push(format!("W+2_{n2}"));
        out.push(format!("W-10_{p1}_{w}"));
        out.push(format!("W0+1_{w}_{n1}"));
        out.push(format!("W-1+1_{p1}_{n1}"));

        // Affixes fire for every word, not only rare ones -- restricting them to
        // out-of-vocabulary words costs eight points against the engine -- but
        // how long they run depends on whether the model has seen the word. A
        // word it knows stops at 4, the classic MXPOST length; letting `PRE1_r`
        // (NN +34.9) and `SUF1_d` (VBD +38.2) compete with ten-letter evidence
        // is what makes "record" a noun in "they record it now". The longer
        // templates are in the model for the words it has never seen.
        let known = !self.weights(&format!("W0_{w}")).is_empty();
        let max = if known { 4 } else { 10 };
        let ch: Vec<char> = w.chars().collect();
        for n in 1..=max.min(ch.len()) {
            let pre: String = ch[..n].iter().collect();
            let suf: String = ch[ch.len() - n..].iter().collect();
            out.push(format!("PRE{n}_{pre}"));
            out.push(format!("SUF{n}_{suf}"));
        }
        if w.contains('-') {
            out.push("CTN_HPN".to_string());
        }
        if w.chars().any(|c| c.is_ascii_digit()) {
            out.push("CTN_NUM".to_string());
        }
    }

    /// Tag every token, lowercasing as the engine does. Callers should only
    /// reach this for an utterance `triggers` accepted.
    pub fn tag_tokens<S: AsRef<str>>(&self, toks: &[S]) -> Vec<String> {
        let lower: Vec<String> = toks.iter().map(|t| t.as_ref().to_lowercase()).collect();
        let mut score = vec![0f32; self.outcomes.len()];
        let mut feats: Vec<String> = Vec::with_capacity(32);
        let mut out = Vec::with_capacity(lower.len());
        for i in 0..lower.len() {
            score.iter_mut().for_each(|s| *s = 0.0);
            self.features(&lower, i, &mut feats);
            for f in &feats {
                let recs = self.weights(f);
                for r in recs.chunks_exact(REC) {
                    score[r[0] as usize] += f32le(r, 1);
                }
            }
            // ties go to the lower outcome index, which is the order the model
            // was packed in and so the order the engine's own table has
            let mut best = 0usize;
            for k in 1..score.len() {
                if score[k] > score[best] {
                    best = k;
                }
            }
            out.push(self.outcomes[best].clone());
        }
        out
    }

    /// Tags for every token of one utterance, or `None` when its words hold no
    /// listed homograph and the engine would not have tagged it.
    ///
    /// `toks` is the utterance as the tagger sees it, punctuation included --
    /// the model is full of features like `W0+1_County_,`, and a capture with
    /// commas beside the homograph agrees 96.6% with punctuation in and 82.4%
    /// with it out. `words` is the subset the gate applies to.
    pub fn tag_tokens_if_triggered<S: AsRef<str>, T: AsRef<str>>(
        &self,
        toks: &[S],
        words: &[T],
    ) -> Option<Vec<String>> {
        if !self.triggers(words) {
            return None;
        }
        Some(self.tag_tokens(toks))
    }

    /// Tags for one utterance given only its words, assuming a closing full
    /// stop. `tag_tokens_if_triggered` is the better entry point where the
    /// utterance's real punctuation is still to hand.
    pub fn tag_utterance<S: AsRef<str>>(&self, words: &[S]) -> Option<Vec<String>> {
        if !self.triggers(words) {
            return None;
        }
        let mut toks: Vec<String> = words.iter().map(|w| w.as_ref().to_string()).collect();
        toks.push(".".to_string());
        let mut tags = self.tag_tokens(&toks);
        tags.pop();
        Some(tags)
    }
}
