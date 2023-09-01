use std::error::Error;


fn main() -> Result<(), Box<dyn Error>> {
    let xaa = std::fs::read("model/xaa")?;
    let xab = std::fs::read("model/xab")?;
    let xac = std::fs::read("model/xac")?;
    let xad = std::fs::read("model/xad")?;
    let xae = std::fs::read("model/xae")?;
    let xaf = std::fs::read("model/xaf")?;
    let xag = std::fs::read("model/xag")?;
    let xah = std::fs::read("model/xah")?;
    let xai = std::fs::read("model/xai")?;
    
    let data = [xaa, xab, xac, xad, xae, xaf, xag, xah, xai].concat();
    std::fs::write("model/model.onnx", data)?; 

    Ok(())
}
