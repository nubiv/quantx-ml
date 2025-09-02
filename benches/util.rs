pub fn scalar_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    out
}

pub fn gen_input(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..len).map(|i| 8.1239412 + (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..len).map(|i| 9.0003 + (i as f32) * 0.002).collect();
    (a, b)
}

// ---------------- x86 paths (Intel mac or Linux/Windows on x86_64) ----------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn sse_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::{_mm_loadu_ps, _mm_mul_ps, _mm_storeu_ps};

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = vec![0.0f32; n];

    let chunks = n / 4;
    let tail = n % 4;

    for k in 0..chunks {
        let idx = k * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));
        let vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(out.as_mut_ptr().add(idx), vc);
    }
    let base = chunks * 4;
    for i in 0..tail {
        out[base + i] = a[base + i] * b[base + i];
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = vec![0.0f32; n];

    let chunks = n / 8;
    let tail = n % 8;

    for k in 0..chunks {
        let idx = k * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), vc);
    }
    let base = chunks * 8;
    for i in 0..tail {
        out[base + i] = a[base + i] * b[base + i];
    }
    out
}

// ---------------- ARM/NEON path (Apple Silicon) ----------------
#[cfg(target_arch = "aarch64")]
pub unsafe fn neon_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::{vld1q_f32, vmulq_f32, vst1q_f32};
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = vec![0.0f32; n];

    let chunks = n / 4;
    let tail = n % 4;

    for k in 0..chunks {
        unsafe {
            let idx = k * 4;
            let va = vld1q_f32(a.as_ptr().add(idx));
            let vb = vld1q_f32(b.as_ptr().add(idx));
            let vc = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(idx), vc);
        }
    }
    let base = chunks * 4;
    for i in 0..tail {
        out[base + i] = a[base + i] * b[base + i];
    }
    out
}
