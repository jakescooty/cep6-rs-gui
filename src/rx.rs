//! A small backtracking regex matcher for the tokenizer's patterns.
//!
//! ceplang_en.dll tests tokens against about forty precompiled Henry Spencer
//! programs. Writing them here in the same notation, rather than as hand-rolled
//! character loops, keeps each rule checkable against `_fe/t2w_cepstral.md`.
//! Supported: literals, `\` escapes, `.`, `[...]` with ranges and `^`, groups
//! with `|`, `?` `*` `+` `{n}` `{n,}` `{n,m}`, and the anchors `^` `$`. A
//! pattern without `^` matches anywhere, as `cst_regex_match` does.

#[derive(Debug)]
enum Node {
    Char(char),
    Any,
    Class(Vec<(char, char)>, bool),
    Group(Vec<Vec<Node>>),
    Repeat(Box<Node>, u32, u32),
    Start,
    End,
}

#[derive(Debug)]
pub struct Rx {
    seq: Vec<Node>,
}

impl Rx {
    pub fn new(pattern: &str) -> Rx {
        let p: Vec<char> = pattern.chars().collect();
        let mut at = 0;
        let alts = parse_alts(&p, &mut at);
        assert!(at == p.len(), "unbalanced ')' in {pattern:?}");
        Rx { seq: vec![Node::Group(alts)] }
    }

    pub fn is_match(&self, s: &str) -> bool {
        let s: Vec<char> = s.chars().collect();
        (0..=s.len()).any(|start| seq_match(&self.seq, &s, start, &|_| true))
    }
}

fn parse_alts(p: &[char], at: &mut usize) -> Vec<Vec<Node>> {
    let mut alts = vec![Vec::new()];
    while *at < p.len() {
        match p[*at] {
            ')' => break,
            '|' => {
                *at += 1;
                alts.push(Vec::new());
            }
            _ => {
                let atom = parse_atom(p, at);
                let atom = parse_quant(p, at, atom);
                alts.last_mut().unwrap().push(atom);
            }
        }
    }
    alts
}

fn parse_atom(p: &[char], at: &mut usize) -> Node {
    let c = p[*at];
    *at += 1;
    match c {
        '^' => Node::Start,
        '$' => Node::End,
        '.' => Node::Any,
        '\\' => {
            let e = p[*at];
            *at += 1;
            Node::Char(e)
        }
        '(' => {
            if p.get(*at) == Some(&'?') && p.get(*at + 1) == Some(&':') {
                *at += 2;
            }
            let alts = parse_alts(p, at);
            assert!(p.get(*at) == Some(&')'), "unclosed group");
            *at += 1;
            Node::Group(alts)
        }
        '[' => {
            let neg = p.get(*at) == Some(&'^');
            if neg {
                *at += 1;
            }
            let mut ranges = Vec::new();
            let mut first = true;
            while p[*at] != ']' || first {
                first = false;
                let a = p[*at];
                *at += 1;
                if p[*at] == '-' && p[*at + 1] != ']' {
                    ranges.push((a, p[*at + 1]));
                    *at += 2;
                } else {
                    ranges.push((a, a));
                }
            }
            *at += 1;
            Node::Class(ranges, neg)
        }
        c => Node::Char(c),
    }
}

fn parse_quant(p: &[char], at: &mut usize, atom: Node) -> Node {
    let (min, max) = match p.get(*at) {
        Some('?') => (0, 1),
        Some('*') => (0, u32::MAX),
        Some('+') => (1, u32::MAX),
        Some('{') => {
            let close = (*at..p.len()).find(|&k| p[k] == '}').expect("unclosed {");
            let body: String = p[*at + 1..close].iter().collect();
            let (lo, hi) = match body.split_once(',') {
                Some((lo, "")) => (lo.parse().unwrap(), u32::MAX),
                Some((lo, hi)) => (lo.parse().unwrap(), hi.parse().unwrap()),
                None => {
                    let n = body.parse().unwrap();
                    (n, n)
                }
            };
            *at = close;
            (lo, hi)
        }
        _ => return atom,
    };
    *at += 1;
    Node::Repeat(Box::new(atom), min, max)
}

fn seq_match(seq: &[Node], s: &[char], pos: usize, k: &dyn Fn(usize) -> bool) -> bool {
    match seq.split_first() {
        None => k(pos),
        Some((n, rest)) => node_match(n, s, pos, &|p| seq_match(rest, s, p, k)),
    }
}

fn node_match(n: &Node, s: &[char], pos: usize, k: &dyn Fn(usize) -> bool) -> bool {
    match n {
        Node::Char(c) => pos < s.len() && s[pos] == *c && k(pos + 1),
        Node::Any => pos < s.len() && k(pos + 1),
        Node::Class(r, neg) => {
            pos < s.len()
                && r.iter().any(|&(a, b)| a <= s[pos] && s[pos] <= b) != *neg
                && k(pos + 1)
        }
        Node::Start => pos == 0 && k(pos),
        Node::End => pos == s.len() && k(pos),
        Node::Group(alts) => alts.iter().any(|a| seq_match(a, s, pos, k)),
        Node::Repeat(inner, min, max) => repeat(inner, *min, *max, 0, s, pos, k),
    }
}

// Greedy: one more repetition first, then the rest of the pattern. An
// iteration that consumes nothing stops the loop.
fn repeat(inner: &Node, min: u32, max: u32, count: u32, s: &[char], pos: usize,
          k: &dyn Fn(usize) -> bool) -> bool
{
    if count < max
        && node_match(inner, s, pos, &|p| p != pos && repeat(inner, min, max, count + 1, s, p, k))
    {
        return true;
    }
    count >= min && k(pos)
}

/// `rx!("^[0-9]+$").is_match(s)`, compiled once per call site.
#[macro_export]
macro_rules! rx {
    ($p:expr) => {{
        static R: std::sync::OnceLock<$crate::rx::Rx> = std::sync::OnceLock::new();
        R.get_or_init(|| $crate::rx::Rx::new($p))
    }};
}

#[cfg(test)]
mod tests {
    use super::Rx;

    #[test]
    fn patterns_from_the_dll() {
        let phone = Rx::new(r"^(1[.-]?)?(\(?[0-9]{3}[)./-]+)?[0-9]{3}[.-][0-9]{4}$");
        assert!(phone.is_match("555-1234"));
        assert!(phone.is_match("1-800-555-1234"));
        assert!(phone.is_match("(555)555-1234"));
        assert!(!phone.is_match("555-12345"));
        let double = Rx::new(r"^-?(([0-9]+\.[0-9]*)|([0-9]+)|(\.[0-9]+))([eE][---+]?[0-9]+)?$");
        assert!(double.is_match("-5") && double.is_match("3.") && double.is_match(".5"));
        assert!(double.is_match("1e-5") && !double.is_match("1.2.3"));
        let unit = Rx::new("[.0-9]+-*[a-zà-öø-ÿ]+");
        assert!(unit.is_match("5km") && unit.is_match("x5-km") && !unit.is_match("km"));
        let roman = Rx::new("^(I{1,3}|IV|VI{0,3}|IX|X[IVX]*)$");
        assert!(roman.is_match("VIII") && roman.is_match("XIV") && !roman.is_match("IL"));
        let money = Rx::new(r"^[$£€][,0-9]+(\.[0-9]+)?$");
        assert!(money.is_match("$1,250.50") && money.is_match("€5") && !money.is_match("$"));
    }
}
