//! Text normalisation: the engine's own, not a set of English rules.
//!
//! `normalize` is a thin driver over `crate::token`, which ports flite's
//! `us_tokentowords` and reads ceplang_en.dll's two decision trees. The rules
//! there are frequently not the conventional readings -- "2017" is "two thousand
//! seventeen" and "10:00" drops the minutes entirely -- and every one of them is
//! checked against the engine rather than chosen.
//!
//! `cardinal` and `ordinal` are kept for callers that want a number spelled out
//! directly. They are not on the synthesis path.
//!
//! `DigitStyle` is the one thing here the engine has no equivalent of. The
//! number tree decides "911" from its neighbours -- "nine hundred eleven" in
//! `The number is 911.`, "nine one one" in `A 911 telephone outage emergency.`
//! -- and `DigitStyle::Digits` overrides it to the code reading for a bare two-
//! or three-digit token, which is what alerting text wants.

use crate::lex::Token;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
    (100, "hundred"),
];
const ORDINAL_ONES: [&str; 20] = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
];
const ORDINAL_TENS: [&str; 10] = [
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
    "seventieth", "eightieth", "ninetieth",
];

/// How to read a run of digits that could be a quantity or a code.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DigitStyle {
    /// 911 is "nine hundred eleven". Right for prose.
    #[default]
    Cardinal,
    /// 911 is "nine one one". Right for codes, phone numbers and alerting text.
    Digits,
}

pub fn cardinal(n: u64, out: &mut Vec<String>) {
    if n < 20 {
        out.push(ONES[n as usize].into());
        return;
    }
    if n < 100 {
        let (t, r) = (n / 10, n % 10);
        if r == 0 {
            out.push(TENS[t as usize].into());
        } else {
            out.push(format!("{}-{}", TENS[t as usize], ONES[r as usize]));
        }
        return;
    }
    for (scale, name) in SCALES {
        if n >= scale {
            cardinal(n / scale, out);
            out.push(name.into());
            let r = n % scale;
            if r > 0 {
                cardinal(r, out);
            }
            return;
        }
    }
}

pub fn ordinal(n: u64, out: &mut Vec<String>) {
    if n < 20 {
        out.push(ORDINAL_ONES[n as usize].into());
        return;
    }
    if n < 100 {
        let (t, r) = (n / 10, n % 10);
        if r == 0 {
            out.push(ORDINAL_TENS[t as usize].into());
        } else {
            out.push(format!("{}-{}", TENS[t as usize], ORDINAL_ONES[r as usize]));
        }
        return;
    }
    // "one hundred first": only the final group becomes ordinal
    let mut head = Vec::new();
    for (scale, name) in SCALES {
        if n >= scale {
            cardinal(n / scale, &mut head);
            head.push(name.into());
            let r = n % scale;
            out.append(&mut head);
            if r > 0 {
                ordinal(r, out);
            } else {
                let last = out.pop().unwrap();
                out.push(format!("{last}th"));
            }
            return;
        }
    }
}

/// Normalise text into the token stream the lexicon expects.
///
/// Without a dictionary to consult, every all-capitals token is spelled out. That
/// is the conservative reading: "n a s a" is merely odd, while a wrong guess at an
/// unknown acronym is unintelligible.
pub fn normalize(text: &str, style: DigitStyle) -> Vec<Token> {
    normalize_with(text, style, |_| false)
}

/// Normalise, consulting `known` for whether a lowercased token is a real word.
/// Pass the lexicon here so acronyms resolve the way the dictionary sees them.
pub fn normalize_with<F: Fn(&str) -> bool>(
    text: &str,
    style: DigitStyle,
    known: F,
) -> Vec<Token> {
    normalize_engine(text, &known, style == DigitStyle::Digits)
}

/// The engine's own path: flite's tokenizer and `us_tokentowords`, with
/// ceplang_en.dll's two decision trees. See `crate::token`.
///
/// A hard break is a sentence end, which the engine renders as two pause units
/// because it starts a new utterance; a phrase break inside one is a single
/// pause.
fn normalize_engine<F: Fn(&str) -> bool>(text: &str, known: &F, digits: bool) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let utts = crate::token::utterances(crate::token::tokenize(text));
    let n = utts.len();
    for (u, toks) in utts.iter().enumerate() {
        for w in crate::token::utterance_words(toks, &|s: &str| known(s), digits) {
            // "prince-edward" reaches the lexicon whole, as the engine sends it
            out.push(Token::Word(w.text.clone()));
            if w.brk {
                out.push(Token::Break(false, w.punc.clone()));
            }
        }
        // the mark that ended the utterance is on the break we are about to
        // drop; the tagger wants it even though the phone stream does not
        let mut mark = String::new();
        while let Some(Token::Break(_, m)) = out.last() {
            if !m.is_empty() {
                mark = m.clone();
            }
            out.pop();
        }
        // Nothing is pushed after the last utterance: a trailing break would add
        // pauses the engine does not make. Its closing mark is therefore the one
        // the tagger never sees, and `PosTag::tag_utterance` assumes a full stop.
        if u + 1 < n {
            out.push(Token::Break(true, mark));
        }
    }
    out
}
