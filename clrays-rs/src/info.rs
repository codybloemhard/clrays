use stopwatch::{Stopwatch};

pub struct Info{
    pub textures: Vec<(String,u64)>,
    pub meta_size: u64,
    pub scene_size: u64,
    pub int_buffer_size: u64,
    pub float_buffer_size: u64,
    times: Vec<(String, i64)>,
    watch: Stopwatch,
}

impl Info{
    pub fn new() -> Self{
        Self{
            textures: Vec::new(),
            meta_size: 0,
            scene_size: 0,
            int_buffer_size: 0,
            float_buffer_size: 0,
            times: Vec::new(),
            watch: Stopwatch::new(),
        }
    }
    pub fn print_info(&self){
        println!("Metadata: {} B.", self.meta_size);
        println!("Scene:    {} B.", self.scene_size);
        println!("Int FB:   {} B.", self.int_buffer_size);
        println!("Float FB: {} B.", self.float_buffer_size);
        let mut sum = 0;
        for (i,(name,size)) in self.textures.iter().enumerate(){
            sum += size;
            println!("Texture{} : {} : {} B.", i, name, size);
        }
        println!("Totalsize: ");
        Self::print_size_verbose(sum);
        println!("Grand Total: ");
        sum += self.meta_size + self.scene_size + self.int_buffer_size + self.float_buffer_size;
        Self::print_size_verbose(sum);
        let mut last = 0;
        for (name,time) in self.times.iter(){
            let elapsed = time - last;
            last = *time;
            println!("{}: {} ms.", name, elapsed);
        }
        println!("Total: {} ms.", last);
    }

    pub fn print_size_verbose(size: u64){
        println!("      {} B.", size);
        println!("      {} KB.", size / 1024);
        println!("      {} MB.", size / 1024u64.pow(2));
        println!("      {} GB.", size / 1024u64.pow(3));
    }

    pub fn start_time(&mut self){
        self.watch.start();
    }

    pub fn set_time_point(&mut self, name: &str){
        self.times.push((name.to_string(),self.watch.elapsed_ms()));
    }
}
