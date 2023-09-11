#[derive(PartialEq, Clone)]
pub enum OptionType {
    TfIdf,
    Vector,
    Bm25,
}

#[derive(Clone)]
pub struct QueryOption {
    options: Vec<OptionType>,
}

impl QueryOption {
    pub fn new() -> Self {
        QueryOption {
            options: Vec::new(),
        }
    }

    pub fn add(&mut self, option: OptionType) -> &mut Self {
        if !self.options.contains(&option) {
            self.options.push(option);
        }
        self
    }

    pub fn build(&mut self) -> Self {
        self.clone()
    }

    pub fn get_options(&self) -> &Vec<OptionType> {
        &self.options
    }
}

impl Default for QueryOption {
    fn default() -> Self {
        QueryOption::new().add(OptionType::TfIdf).build()
    }
}
