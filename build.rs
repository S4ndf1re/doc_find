use std::error::Error;


fn main() -> Result<(), Box<dyn Error>> {
    let xaa = std::fs::read("model/xaa")?;
    let xab = std::fs::read("model/xab")?;
    
    let data = [xaa, xab].concat();
    std::fs::write("model/model.onnx", data)?; 

    Ok(())
}
