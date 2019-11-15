pub trait State{
    fn new() -> Self;
    fn should_close(&self) -> bool;
    fn set_to_close(&mut self);
    fn update(&mut self, dt: f64);
}

pub struct StdState{
    should_close: bool,
}

impl State for StdState{
    fn new() -> Self{
        StdState{ should_close: false }
    }

    fn should_close(&self) -> bool{
        self.should_close
    }

    fn set_to_close(&mut self){
        self.should_close = true;
    }

    fn update(&mut self, dt: f64){ }
}
