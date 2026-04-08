fn main() {
    if std::env::var_os("CARGO_FEATURE_PQCLEAN_REF").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tests/pqclean_ref");

    let mut build = cc::Build::new();
    build
        .include("tests/pqclean_ref")
        .flag_if_supported("-std=c99")
        .warnings(false)
        .file("tests/pqclean_ref/fips202.c")
        .file("tests/pqclean_ref/falcon-512/codec.c")
        .file("tests/pqclean_ref/falcon-512/fft.c")
        .file("tests/pqclean_ref/falcon-512/fpr.c")
        .file("tests/pqclean_ref/falcon-512/keygen.c")
        .file("tests/pqclean_ref/falcon-512/rng.c")
        .file("tests/pqclean_ref/falcon-512/vrfy.c")
        .file("tests/pqclean_ref/falcon-1024/codec.c")
        .file("tests/pqclean_ref/falcon-1024/fft.c")
        .file("tests/pqclean_ref/falcon-1024/fpr.c")
        .file("tests/pqclean_ref/falcon-1024/keygen.c")
        .file("tests/pqclean_ref/falcon-1024/rng.c")
        .file("tests/pqclean_ref/falcon-1024/vrfy.c")
        .file("tests/pqclean_ref/oracle_512.c")
        .file("tests/pqclean_ref/oracle_1024.c")
        .compile("pqclean_ref");
}
