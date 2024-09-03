cargo build --release
./target/release/params 15 kzg 
wait
./target/release/params 17 kzg 
wait
./target/release/params 26 kzg &> params_kzg_26.txt &
wait
./target/release/params 27 kzg &> params_kzg_27.txt &
wait 
./target/release/params 26 ipa &> params_kzg_26.txt &
wait 
./target/release/params 27 ipa &> params_kzg_27.txt &
wait