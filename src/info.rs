use stopwatch::{ Stopwatch };

#[derive(Default)]
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
        Self::default()
    }

    pub fn print_info(&self){
        println!("Metadata: {}.", Self::format_size(self.meta_size));
        println!("Scene:    {}.", Self::format_size(self.scene_size));
        println!("Int FB:   {}.", Self::format_size(self.int_buffer_size));
        println!("Float FB: {}.", Self::format_size(self.float_buffer_size));
        let mut sum = 0;
        for (i, (name, size)) in self.textures.iter().enumerate(){
            sum += size;
            println!("{} : {} : {}.", i, name, Self::format_size(*size));
        }
        println!("Totalsize: ");
        Self::print_size_verbose(sum);
        println!("Grand Total: ");
        sum += self.meta_size + self.scene_size + self.int_buffer_size + self.float_buffer_size;
        Self::print_size_verbose(sum);
        let mut last = 0;
        for (name, time) in self.times.iter(){
            let elapsed = time - last;
            last = *time;
            println!("{}: {} ms.", name, elapsed);
        }
        println!("Total: {} ms.", last);
    }

    pub fn print_size_verbose(size: u64){
        println!("\t{} B.", size);
        println!("\t{} KB.", size / 1024);
        println!("\t{} MB.", size / 1024u64.pow(2));
        println!("\t{} GB.", size / 1024u64.pow(3));
    }

    pub fn format_size(size: u64) -> String{
        let mut max = 1_u64;
        for label in ["B", "KB", "MB", "GB"]{
            let new_max = max * 1024;
            if size < new_max {
                return format!("{} {}", size / max, label);
            }
            max = new_max;
        }
        return format!("{} {}", size / max, "TB");
    }

    pub fn start_time(&mut self){
        self.watch.start();
    }

    pub fn stop_time(&mut self){
        self.watch.stop();
    }

    pub fn set_time_point(&mut self, name: &str){
        self.times.push((name.to_string(), self.watch.elapsed_ms()));
    }
}
