
#[derive(Debug)]
pub struct Error {
    error_message: String,
}

impl Error {
    pub fn new(error_message: String) -> Error {
        Error {
            error_message: error_message,
        }
    }
}
