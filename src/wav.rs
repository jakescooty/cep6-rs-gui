//! Minimal 16-bit mono RIFF writer.

use std::io::{self, Write};
use std::path::Path;

pub fn write(path: impl AsRef<Path>, pcm: &[i16], sample_rate: u32) -> io::Result<()> {
    let data_len = (pcm.len() * 2) as u32;
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;              // PCM
    f.write_all(&1u16.to_le_bytes())?;              // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;              // block align
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in pcm {
        f.write_all(&s.to_le_bytes())?;
    }
    f.flush()
}

/// A WAV being written a piece at a time, for input too long to hold as PCM.
///
/// A RIFF header has to state two lengths it cannot know until the end, so the
/// header goes down with zeroes and `finish` seeks back and patches them. Drop
/// without `finish` and the file is left claiming no samples.
pub struct Writer {
    f: io::BufWriter<std::fs::File>,
    samples: u64,
}

impl Writer {
    pub fn create(path: impl AsRef<Path>, sample_rate: u32) -> io::Result<Writer> {
        let mut f = io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"RIFF")?;
        f.write_all(&0u32.to_le_bytes())?;              // patched by finish
        f.write_all(b"WAVEfmt ")?;
        f.write_all(&16u32.to_le_bytes())?;
        f.write_all(&1u16.to_le_bytes())?;
        f.write_all(&1u16.to_le_bytes())?;
        f.write_all(&sample_rate.to_le_bytes())?;
        f.write_all(&(sample_rate * 2).to_le_bytes())?;
        f.write_all(&2u16.to_le_bytes())?;
        f.write_all(&16u16.to_le_bytes())?;
        f.write_all(b"data")?;
        f.write_all(&0u32.to_le_bytes())?;              // patched by finish
        Ok(Writer { f, samples: 0 })
    }

    pub fn push(&mut self, pcm: &[i16]) -> io::Result<()> {
        let mut buf = Vec::with_capacity(pcm.len() * 2);
        for &s in pcm {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        self.f.write_all(&buf)?;
        self.samples += pcm.len() as u64;
        Ok(())
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Patch the two lengths and close. A RIFF length is 32 bits, so anything
    /// past 4 GB of audio is saturated rather than wrapped to a smaller number
    /// that would make the file look truncated.
    pub fn finish(self) -> io::Result<u64> {
        use io::{Seek, SeekFrom};
        let n = self.samples;
        let data_len = u32::try_from(n * 2).unwrap_or(u32::MAX);
        let mut f = self.f.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(4))?;
        f.write_all(&data_len.saturating_add(36).to_le_bytes())?;
        f.seek(SeekFrom::Start(40))?;
        f.write_all(&data_len.to_le_bytes())?;
        f.flush()?;
        Ok(n)
    }
}

/// Read a 16-bit mono PCM WAV. Returns the samples and the sample rate.
pub fn read<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<(Vec<i16>, u32)> {
    let b = std::fs::read(path)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if b.len() < 44 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);

    let (mut sps, mut bits, mut chans) = (0u32, 0u16, 0u16);
    let mut data: Option<(usize, usize)> = None;
    let mut o = 12usize;
    while o + 8 <= b.len() {
        let id = &b[o..o + 4];
        let len = u32at(o + 4) as usize;
        let body = o + 8;
        if body + len > b.len() {
            break;
        }
        if id == b"fmt " && len >= 16 {
            chans = u16at(body + 2);
            sps = u32at(body + 4);
            bits = u16at(body + 14);
        } else if id == b"data" {
            data = Some((body, len));
        }
        o = body + len + (len & 1);
    }
    let Some((off, len)) = data else { return Err(bad("no data chunk")) };
    if bits != 16 {
        return Err(bad("only 16-bit PCM is supported"));
    }
    if chans != 1 {
        return Err(bad("only mono is supported"));
    }
    let n = len / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(i16::from_le_bytes([b[off + 2 * i], b[off + 2 * i + 1]]));
    }
    Ok((out, sps))
}
