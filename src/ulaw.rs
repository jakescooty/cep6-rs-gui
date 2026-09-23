//! G.711 mu-law decode table, identical to the one in swift.dll at RVA 0x79350.

pub static DECODE: [i16; 256] = build();

const fn build() -> [i16; 256] {
    let mut t = [0i16; 256];
    let mut i = 0usize;
    while i < 256 {
        let u = (!(i as u8)) as i32;
        let man = u & 0x0F;
        let exp = (u >> 4) & 0x07;
        let mut v = (((man << 1) + 33) << exp) - 33;
        v <<= 2;
        t[i] = if u & 0x80 != 0 { -v as i16 } else { v as i16 };
        i += 1;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::DECODE;

    #[test]
    fn matches_g711() {
        assert_eq!(DECODE[0], -32124);
        assert_eq!(DECODE[0x7F], -0);
        assert_eq!(DECODE[0x80], 32124);
        assert_eq!(DECODE[0xFF], 0);
    }
}
