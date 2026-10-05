//! Fixed-size bitset over slice positions (u64 words), C-order (position i -> word i/64).

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Bits {
    pub w: Vec<u64>,
}

impl Bits {
    pub fn new(nbits: usize) -> Self {
        Bits {
            w: vec![0u64; (nbits + 63) / 64],
        }
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        self.w[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.w[i >> 6] >> (i & 63)) & 1 != 0
    }

    #[inline]
    pub fn or_with(&mut self, o: &Bits) {
        for (a, b) in self.w.iter_mut().zip(o.w.iter()) {
            *a |= *b;
        }
    }

    #[inline]
    pub fn and_with(&mut self, o: &Bits) {
        for (a, b) in self.w.iter_mut().zip(o.w.iter()) {
            *a &= *b;
        }
    }

    pub fn or2(a: &Bits, b: &Bits) -> Bits {
        let mut r = a.clone();
        r.or_with(b);
        r
    }

    pub fn and2(a: &Bits, b: &Bits) -> Bits {
        let mut r = a.clone();
        r.and_with(b);
        r
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.w.iter().map(|x| x.count_ones() as u64).sum()
    }
}
