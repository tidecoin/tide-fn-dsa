use tide_fn_dsa_comm::shake;

#[derive(Copy, Clone, Debug)]
pub(crate) struct SHAKE256x4 {
    sh: [shake::SHAKE256; 4],
    buf: [u8; 4 * 136],
    ptr: usize,
}

impl SHAKE256x4 {
    pub(crate) fn new(seed: &[u8]) -> Self {
        let mut sh = [
            shake::SHAKE256::new(),
            shake::SHAKE256::new(),
            shake::SHAKE256::new(),
            shake::SHAKE256::new(),
        ];
        for (i, shi) in sh.iter_mut().enumerate() {
            shi.inject(seed).unwrap();
            shi.inject(&[i as u8]).unwrap();
            shi.flip().unwrap();
        }
        Self {
            sh,
            buf: [0u8; 4 * 136],
            ptr: 4 * 136,
        }
    }

    fn refill(&mut self) {
        self.ptr = 0;
        for i in 0..(4 * 136 / 32) {
            for j in 0..4 {
                let k = 32 * i + 8 * j;
                self.sh[j].extract(&mut self.buf[k..(k + 8)]).unwrap();
            }
        }
    }

    pub(crate) fn next_u16(&mut self) -> u16 {
        if self.ptr >= 4 * 136 - 1 {
            self.refill();
        }
        let x = u16::from_le_bytes(
            *<&[u8; 2]>::try_from(&self.buf[self.ptr..self.ptr + 2]).unwrap(),
        );
        self.ptr += 2;
        x
    }

    #[allow(dead_code)]
    pub(crate) fn next_u64(&mut self) -> u64 {
        if self.ptr >= 4 * 136 - 7 {
            self.refill();
        }
        let x = u64::from_le_bytes(
            *<&[u8; 8]>::try_from(&self.buf[self.ptr..self.ptr + 8]).unwrap(),
        );
        self.ptr += 8;
        x
    }
}
