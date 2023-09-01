use std::error::Error;


fn main() -> Result<(), Box<dyn Error>> {
    let xaa = std::fs::read("model/xaa")?;
    let xab = std::fs::read("model/xab")?;
    let xac = std::fs::read("model/xac")?;
    let xad = std::fs::read("model/xad")?;
    let xae = std::fs::read("model/xae")?;
    
    let data = [xaa, xab, xac, xad, xae].concat();
    std::fs::write("model/model.onnx", data)?; 

    Ok(())
}
