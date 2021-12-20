// imports
// use core::simd::*;
// use arrayvec::ArrayVec;


// import stopwatch
use std::default::Default;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct Stopwatch {
	/// The time the stopwatch was started last, if ever.
	start_time: Option<Instant>,
	/// The time the stopwatch was split last, if ever.
	split_time: Option<Instant>,
	/// The time elapsed while the stopwatch was running (between start() and stop()).
	elapsed: Duration,
}

impl Default for Stopwatch {
	fn default() -> Stopwatch {
		Stopwatch {
			start_time: None,
			split_time: None,
			elapsed: Duration::from_secs(0),
		}
	}
}

impl fmt::Display for Stopwatch {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		return write!(f, "{}ms", self.elapsed_ms());
	}
}

impl Stopwatch {
	/// Returns a new stopwatch.
	pub fn new() -> Stopwatch {
		let sw: Stopwatch = Default::default();
		return sw;
	}

	/// Returns a new stopwatch which will immediately be started.
	pub fn start_new() -> Stopwatch {
		let mut sw = Stopwatch::new();
		sw.start();
		return sw;
	}

	/// Starts the stopwatch.
	pub fn start(&mut self) {
		self.start_time = Some(Instant::now());
	}

	/// Stops the stopwatch.
	pub fn stop(&mut self) {
		self.elapsed = self.elapsed();
		self.start_time = None;
		self.split_time = None;
	}

	/// Resets all counters and stops the stopwatch.
	pub fn reset(&mut self) {
		self.elapsed = Duration::from_secs(0);
		self.start_time = None;
		self.split_time = None;
	}

	/// Resets and starts the stopwatch again.
	pub fn restart(&mut self) {
		self.reset();
		self.start();
	}

	/// Returns whether the stopwatch is running.
	pub fn is_running(&self) -> bool {
		return self.start_time.is_some();
	}

	/// Returns the elapsed time since the start of the stopwatch.
	pub fn elapsed(&self) -> Duration {
		match self.start_time {
			// stopwatch is running
			Some(t1) => {
				return t1.elapsed() + self.elapsed;
			}
			// stopwatch is not running
			None => {
				return self.elapsed;
			}
		}
	}

	/// Returns the elapsed time since the start of the stopwatch in milliseconds.
	pub fn elapsed_ms(&self) -> i64 {
		let dur = self.elapsed();
		return (dur.as_secs() * 1000 + (dur.subsec_nanos() / 1000000) as u64) as i64;
	}

	/// Returns the elapsed time since last split or start/restart.
	///
	/// If the stopwatch is in stopped state this will always return a zero Duration.
	pub fn elapsed_split(&mut self) -> Duration {
		match self.start_time {
			// stopwatch is running
			Some(start) => {
				let res = match self.split_time {
					Some(split) => split.elapsed(),
					None => start.elapsed(),
				};
				self.split_time = Some(Instant::now());
				res
			}
			// stopwatch is not running
			None => Duration::from_secs(0),
		}
	}

	/// Returns the elapsed time since last split or start/restart in milliseconds.
	///
	/// If the stopwatch is in stopped state this will always return zero.
	pub fn elapsed_split_ms(&mut self) -> i64 {
		let dur = self.elapsed_split();
		return (dur.as_secs() * 1000 + (dur.subsec_nanos() / 1_000_000) as u64) as i64;
	}
}



// consts
pub const GAMMA: f32 = 2.2;
pub const PI: f32 = std::f32::consts::PI;
pub const FRAC_2_PI: f32 = 0.5 * std::f32::consts::PI;
pub const FRAC_4_PI: f32 = 0.25 * std::f32::consts::PI;
pub const MAX_RENDER_DIST: f32 = 1000000.0;
pub const EPSILON: f32 = 0.001;

// Scene
pub trait Bufferizable{
    fn get_data(&self) -> Vec<f32>;
}
#[derive(Default,Debug)]
pub struct Triangle{ // 37 byte
    pub a: Vec3, // Vec3: 12 byte
    pub b: Vec3, // Vec3: 12 byte
    pub c: Vec3, // Vec3: 12 byte
}
impl Bufferizable for Triangle {
    fn get_data(&self) -> Vec<f32> {
        vec![
            self.a.x, self.a.y, self.a.z,
            self.b.x, self.b.y, self.b.z,
            self.c.x, self.c.y, self.c.z,
        ]
    }
}

// Vec3
#[derive(Clone, Copy, Debug)]
pub struct Orientation {
    pub yaw: f32,
    pub roll: f32
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3{ // 12 byte
    pub x: f32, // f32: 4 byte
    pub y: f32, // f32: 4 byte
    pub z: f32, // f32: 4 byte
}
impl PartialEq for Vec3{
    fn eq(&self, other: &Self) -> bool {
        self.dist(*other) < EPSILON
    }
}
impl Eq for Vec3 {}

impl Bufferizable for Vec3{
    fn get_data(&self) -> Vec<f32>{
        vec![ self.x, self.y, self.z ]
    }
}

impl Vec3{
    pub const ZERO: Vec3 =      Self { x:  0.0, y:  0.0, z:  0.0 };
    pub const ONE: Vec3 =       Self { x:  1.0, y:  1.0, z:  1.0 };
    pub const LEFT: Vec3 =      Self { x:  1.0, y:  0.0, z:  0.0 };
    pub const RIGHT: Vec3 =     Self { x: -1.0, y:  0.0, z:  0.0 };
    pub const UP: Vec3 =        Self { x:  0.0, y:  1.0, z:  0.0 };
    pub const DOWN: Vec3 =      Self { x:  0.0, y: -1.0, z:  0.0 };
    pub const FORWARD: Vec3 =   Self { x:  0.0, y:  0.0, z:  1.0 };
    pub const BACKWARD: Vec3 =  Self { x:  0.0, y:  0.0, z: -1.0 };
    pub const RED: Vec3 =       Self { x:  1.0, y:  0.0, z:  0.0 };
    pub const GREEN: Vec3 =     Self { x:  0.0, y:  1.0, z:  0.0 };
    pub const BLUE: Vec3 =      Self { x:  0.0, y:  0.0, z:  1.0 };
    pub const BLACK: Vec3 =     Self { x:  0.0, y:  0.0, z:  0.0 };
    pub const WHITE: Vec3 =     Self { x:  1.0, y:  1.0, z:  1.0 };
    pub const EPSILON: Vec3 =   Self { x: EPSILON, y: EPSILON, z: EPSILON};

    #[inline]
    pub fn as_array(&self) -> [f32;3]{ [self.x,self.y,self.z] }
    pub fn into_arr(&self) -> [f32;3]{ [self.x,self.y,self.z] }

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self{
        Self { x, y, z }
    }

    #[inline]
    pub fn uni(v: f32) -> Self{
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn from_orientation(ori: &Orientation) -> Self {
        let a = ori.roll;  // Up/Down
        let b = ori.yaw;   // Left/Right
        Self { x: a.cos() * b.sin(), y: a.sin(), z: a.cos() * -b.cos() }
    }


    #[inline]
    pub fn yawed(self, o: f32) -> Self {
        Self {
            x: self.x * o.cos() + self.z * o.sin(),
            y: self.y,
            z: -self.x * o.sin() + self.z * o.cos()
        }
    }
    #[inline]
    pub fn fake_arr(self, axis: Axis) -> f32 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }

    #[inline]
    pub fn orientation(&self) -> Orientation {
        let normalized = self.normalized_fast();
        Orientation {
            yaw: f32::atan2(normalized.x,-normalized.z),
            roll: normalized.y.asin()
        }
    }

    #[inline]
    pub fn clamp(&mut self, b: f32, t: f32){
        self.x = self.x.max(b).min(t);
        self.y = self.y.max(b).min(t);
        self.z = self.z.max(b).min(t);
    }

    #[inline]
    pub fn clamped(mut self, b: f32, t: f32) -> Self{
        self.clamp(b, t);
        self
    }

    #[inline]
    pub fn unharden(&mut self, s: f32){
        self.clamp(s, 1.0 - s);
    }

    #[inline]
    pub fn unhardened(mut self, s: f32) -> Self{
        self.unharden(s);
        self
    }

    #[inline]
    pub fn neg(&mut self){
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
    }

    #[inline]
    pub fn neged(mut self) -> Self{
        self.neg();
        self
    }

    #[inline]
    pub fn dot(self, o: Self) -> f32{
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn len(self) -> f32{
        (self.dot(self)).sqrt()
    }

    pub fn dist(self, o: Self) -> f32{
        self.subed(o).len()
    }

    #[inline]
    pub fn normalize_fast(&mut self){
        let l = self.len();
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    #[inline]
    pub fn normalize(&mut self){
        let l = self.len();
        if l == 0.0 { return; }
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    #[inline]
    pub fn normalized_fast(mut self) -> Self{
        self.normalize_fast();
        self
    }

    #[inline]
    pub fn normalized(mut self) -> Self{
        self.normalize();
        self
    }

    #[inline]
    pub fn scale(&mut self, s: f32){
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }

    #[inline]
    pub fn scaled(mut self, s: f32) -> Self{
        self.scale(s);
        self
    }

    #[inline]
    pub fn add_scalar(&mut self, s: f32){
        self.x += s;
        self.y += s;
        self.z += s;
    }

    #[inline]
    pub fn added_scalar(mut self, s: f32) -> Self{
        self.add_scalar(s);
        self
    }

    #[inline]
    pub fn add(&mut self, o: Self){
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }

    #[inline]
    pub fn added(mut self, o: Self) -> Self{
        self.add(o);
        self
    }

    #[inline]
    pub fn sub(&mut self, o: Self){
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }

    #[inline]
    pub fn subed(mut self, o: Self) -> Self{
        self.sub(o);
        self
    }

    #[inline]
    pub fn mul(&mut self, o: Self){
        self.x *= o.x;
        self.y *= o.y;
        self.z *= o.z;
    }

    #[inline]
    pub fn muled(mut self, o: Self) -> Self{
        self.mul(o);
        self
    }

    #[inline]
    pub fn div_fast(&mut self, o: Self){
        self.x /= o.x;
        self.y /= o.y;
        self.z /= o.z;
    }

    #[inline]
    pub fn dived_fast(mut self, o: Self) -> Self{
        self.div_fast(o);
        self
    }

    #[inline]
    pub fn div(&mut self, o: Self){
        if o.x != 0.0 { self.x /= o.x; }
        else { self.x = std::f32::MAX; }
        if o.y != 0.0 { self.y /= o.y; }
        else { self.y = std::f32::MAX; }
        if o.z != 0.0 { self.z /= o.z; }
        else { self.z = std::f32::MAX; }
    }

    #[inline]
    pub fn dived(mut self, o: Self) -> Self{
        self.div(o);
        self
    }

    #[inline]
    pub fn div_scalar_fast(&mut self, s: f32){
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }

    #[inline]
    pub fn dived_scalar_fast(mut self, s: f32) -> Self{
        self.div_scalar_fast(s);
        self
    }

    #[inline]
    pub fn pow_scalar(&mut self, s: f32){
        self.x = self.x.powf(s);
        self.y = self.y.powf(s);
        self.z = self.z.powf(s);
    }

    #[inline]
    pub fn powed_scalar(mut self, s: f32) -> Self{
        self.pow_scalar(s);
        self
    }

    #[inline]
    pub fn cross(&mut self, o: Self){
        let xx = self.y * o.z - self.z * o.y;
        let yy = self.z * o.x - self.x * o.z;
        let zz = self.x * o.y - self.y * o.x;
        self.x = xx;
        self.y = yy;
        self.z = zz;
    }

    #[inline]
    pub fn crossed(self, o: Self) -> Self{
        let x = self.y * o.z - self.z * o.y;
        let y = self.z * o.x - self.x * o.z;
        let z = self.x * o.y - self.y * o.x;
        Self { x, y, z }
    }

    #[inline]
    pub fn sum(self) -> f32{
        self.x + self.y + self.z
    }

    #[inline]
    pub fn reflected(self, nor: Vec3) -> Self{
        Self::subed(self, nor.scaled(2.0 * Self::dot(self, nor)))
    }

    #[inline]
    pub fn mix(&mut self, o: Self, t: f32){
        #[inline]
        fn lerp(a: f32, b: f32, t: f32) -> f32{
            a + t * (b - a)
        }
        self.x = lerp(self.x, o.x, t);
        self.y = lerp(self.y, o.y, t);
        self.z = lerp(self.z, o.z, t);
    }

    #[inline]
    pub fn mixed(mut self, o: Self, t: f32) -> Self{
        self.mix(o, t);
        self
    }

    #[inline]
    pub fn equal(self, other: &Vec3) -> bool{
        (self.x-other.x).abs() < EPSILON &&
        (self.y-other.y).abs() < EPSILON &&
        (self.z-other.z).abs() < EPSILON
    }

    pub fn less_eq(self, o: Self) -> bool{
        self.x <= o.x && self.y <= o.y && self.z <= o.z
    }
}

// AABB
#[derive(Copy, Clone, Debug)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2
}

impl Axis {
    pub fn as_usize(&self) -> usize{
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AABB { // 24 bytes
    pub min: Vec3, // Vec3: 12 bytes
    pub max: Vec3, // Vec3: 12 bytes
}

impl Default for AABB{
    fn default() -> Self{
        Self::new()
    }
}

impl AABB {
    pub fn new() -> Self{
        Self {
            min: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX, },
            max: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN, },
        }
    }

    #[inline]
    pub fn set_default(&mut self) {
        self.max.x = f32::MIN;
        self.max.y = f32::MIN;
        self.max.z = f32::MIN;
        self.min.x = f32::MAX;
        self.min.y = f32::MAX;
        self.min.z = f32::MAX;
    }

    pub fn from_point_radius(p: Vec3, r: f32) -> Self{
        Self{
            min: p.subed(Vec3::uni(r)),
            max: p.added(Vec3::uni(r)),
        }
    }

    pub fn from_points(ps: &[Vec3]) -> Self{
        let (mut minx, mut miny, mut minz): (f32, f32, f32) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut maxx, mut maxy, mut maxz): (f32, f32, f32) = (f32::MIN, f32::MIN, f32::MIN);

        for p in ps{
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            minz = minz.min(p.z);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
            maxz = maxz.max(p.z);
        }

        Self{
            min: Vec3::new(minx, miny, minz),
            max: Vec3::new(maxx, maxy, maxz),
        }
    }

    #[inline]
    pub fn combine(&mut self, other: Self){
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    #[inline]
    pub fn combined(mut self, other: Self) -> Self{
        self.combine(other);
        self
    }

    pub fn grow(&mut self, v: Vec3){
        self.min.sub(v);
        self.max.add(v);
    }

    pub fn grown(mut self, v: Vec3) -> Self{
        self.grow(v);
        self
    }

    #[inline]
    pub fn midpoint(&self) -> Vec3{
        self.lerp(0.5)
    }

    #[inline]
    pub fn lerp(&self, val: f32) -> Vec3{
        self.min.scaled(1.0 - val).added(self.max.scaled(val))
    }

    pub fn volume(self) -> f32{
        let v = self.max.subed(self.min);
        v.x * v.y * v.z
    }

    #[inline]
    pub fn surface_area(self) -> f32{
        let v = self.max.subed(self.min);
        v.x * v.y * 2.0 + v.x * v.z * 2.0 + v.y * v.z * 2.0
    }

    #[inline]
    pub fn is_in(self, other: &AABB) -> bool{
        other.min.less_eq(self.min) && self.max.less_eq(other.max)
    }

    #[inline]
    pub fn is_equal(self, other: &AABB) -> bool{
        self.min.equal(&other.min) && self.max.equal(&other.max)
    }
}

// Bvh
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex{
    pub bound: [f32; 6],
    left_first: usize,
    count: usize,
}

const STACK_SIZE : usize = 100000;
pub struct CustomStack<T> {
    pub stack: [T; STACK_SIZE],
    pub index: usize
}

impl<T: Default + Copy> CustomStack<T> {
    pub fn new() -> Self{
        Self { stack: [T::default(); STACK_SIZE], index: 0 }
    }
    #[inline]
    pub fn current(&self) -> &T{
        &self.stack[self.index]
    }
    pub fn push(&mut self, item: T){
        self.index += 1;
        self.stack[self.index] = item;
    }
    pub fn pop(&mut self) -> T{
        self.index = self.index-1;
        self.stack[self.index + 1]
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct StackItem {
    pub current : usize,
    pub first : usize,
    pub count : usize,
    pub depth : usize
}


// main
#[inline]
fn xor32(seed: &mut u32) -> u32{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

// const TRIANGLES: usize = 20000000;
// const TRIANGLES: usize = 10000000;
const TRIANGLES: usize = 5000000;
// const TRIANGLES: usize = 2500000;
// const TRIANGLES: usize = 1000000;
// const TRIANGLES: usize = 100000;

fn generate_triangles() -> Vec<Triangle>{
    // generate triangles
    let mut triangles = vec![];
    let mut seed:u32 = 81349324; // guaranteed to be random
    for i in 0..TRIANGLES{
        if i % 100000 == 0 {
            println!("{}",i);
        }
        triangles.push(Triangle{
            a: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
            b: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
            c: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
        });
        // println!("{},{:?}",i, triangles[i]);
    }
    triangles

    // // convert to builder data
    // let n = triangles.len();
    // (0..n).into_iter().map(|i|
    //     AABB::from_points(&[triangles[i].a, triangles[i].b, triangles[i].c])
    // ).collect::<Vec<_>>()
}

fn main() {
    let triangles = generate_triangles();
    let n = triangles.len();
    let timer_prepare = Stopwatch::start_new();
    let mut bounds = triangles.into_iter()
        .map(|triangle| AABB::from_points(&[triangle.a, triangle.b, triangle.c]))
        .map(|aabb| [aabb.min.into_arr(), aabb.max.into_arr()])
        .map(|p| [p[0][0],p[0][1],p[0][2],p[1][0],p[1][1],p[1][2]])
        // .flatten()
        // .flatten()
        .collect::<Vec<[f32;6]>>();

    // TEST: are bounds valid?
    for aabb in &bounds {
        assert!(aabb[0] <= aabb[3]);
        assert!(aabb[1] <= aabb[4]);
        assert!(aabb[2] <= aabb[5]);
    }
    // println!("{:?}", bounds.as_slice()[0]);
    // TEST end

    let count = n;
    let mut is = (0..n).into_iter().collect::<Vec<_>>();
    let mut vs = vec![Vertex::default(); n * 2];
    let bins = 12;
    println!("{}", timer_prepare.elapsed_ms());

    // build bvh
    let current = 0;
    let first = 0;
    let mut poolptr = 2;
    let mut depth = 0;

    let binsf = bins as f32;
    let binsf_inf = 1.0 / binsf;

    let mut stack = vec![]; // [(current,first,count,step)]
    stack.push(StackItem {current,first,count,depth});

    let mut lerps = vec![[0.0,0.0,0.0]; bins-1];
    let mut binbounds = vec![[0.0;6];bins];
    let mut bincounts : Vec<usize> = vec![0;bins];

    let mut lb = AABB_NULL;
    let mut rb = AABB_NULL;

    let mut best_aabb_left = AABB_NULL;
    let mut best_aabb_right = AABB_NULL;
    let mut best_axis = 0;
    let mut best_split = 0.0;

    let mut sub_is: &[usize];
    let mut v : &mut Vertex = &mut Vertex::default();
    let mut top_bound : [f32;6];

    let mut current = 0;
    let mut first = 0;
    let mut count = 0;
    let mut step = 0;

    // debug info
    let mut handled = 0;
    let mut counter = 0;
    let mut last_handled = 0;
    // end of debug info

    let mut timer = Stopwatch::start_new();
    let mut depth_timers = vec![];
    let mut depth_counters = vec![];
    let mut depth_items = vec![];

    while stack.len() > 0 {
        // if handled / 100000 > last_handled {
        //     println!("{}", handled);
        //     last_handled = handled / 100000;
        // }
        timer.start();

        let mut x = stack.pop().unwrap();
        // measure time in depth
        depth = x.depth;
        if depth >= depth_timers.len() {
            depth_timers.push(0);
            depth_counters.push(0);
            depth_items.push(0);
        }
        depth_counters[depth] += 1;
        depth_items[depth] += x.count;

        current = x.current;
        count = x.count;
        first = x.first;
        v = &mut vs[current];

        // sub_is = &is[first..first + count];
        let sub_range = first..first + count;
        top_bound = union_bound(&bounds[sub_range.clone()]);
        v.bound = top_bound;

        if count < 3 { // leaf
            handled += count;
            v.left_first = first; // first
            v.count = count;
            continue;
        }

        // sah binned

        // precompute lerps
        for (i, item) in lerps.iter_mut().enumerate(){ // lerp
            item[0] = top_bound[0] + (top_bound[3] - top_bound[0]) * (i+1) as f32 * binsf_inf;
            item[1] = top_bound[1] + (top_bound[4] - top_bound[1]) * (i+1) as f32 * binsf_inf;
            item[2] = top_bound[2] + (top_bound[5] - top_bound[2]) * (i+1) as f32 * binsf_inf;
        }

        // compute best combination; minimal cost
        let (mut ls, mut rs) = (0, 0);
        lb = AABB_NULL;
        rb = AABB_NULL;
        let max_cost = count as f32 * surface_area(top_bound);
        let mut best_cost = max_cost;

        for axis in [0, 1, 2] {

            // if !axis_valid[u] { continue; }
            let k1 = (binsf*(1.0-EPSILON))/(top_bound[3+axis]-top_bound[axis]);
            let k0 = top_bound[axis];

            // place bounds in bins
            // generate bounds of bins
            binbounds.fill(AABB_NULL);
            bincounts.fill(0);
            let mut index: usize ;
            for index_triangle in sub_range.clone() {
                let midpoint = 0.5 * (bounds[index_triangle][axis] + bounds[index_triangle][axis + 3]);
                index = (k1*(midpoint-k0)) as usize;
                // combine
                binbounds[index][0] = binbounds[index][0].min(bounds[index_triangle][0]);
                binbounds[index][1] = binbounds[index][1].min(bounds[index_triangle][1]);
                binbounds[index][2] = binbounds[index][2].min(bounds[index_triangle][2]);
                binbounds[index][3] = binbounds[index][3].max(bounds[index_triangle][3]);
                binbounds[index][4] = binbounds[index][4].max(bounds[index_triangle][4]);
                binbounds[index][5] = binbounds[index][5].max(bounds[index_triangle][5]);
                bincounts[index] += 1;
            }

            // iterate over bins
            for (lerp_index,lerp) in lerps.iter().enumerate(){
                let split = lerp[axis];
                // reset values
                ls = 0;
                rs = 0;
                lb = AABB_NULL;
                rb = AABB_NULL;
                // construct lerpbounds
                for j in 0..lerp_index { // left of split
                    ls += bincounts[j];
                    // combine
                    lb[0] = lb[0].min(binbounds[j][0]);
                    lb[1] = lb[1].min(binbounds[j][1]);
                    lb[2] = lb[2].min(binbounds[j][2]);
                    lb[3] = lb[3].max(binbounds[j][3]);
                    lb[4] = lb[4].max(binbounds[j][4]);
                    lb[5] = lb[5].max(binbounds[j][5]);
                }
                for j in lerp_index..bins { // right of split
                    rs += bincounts[j];
                    // combine
                    rb[0] = rb[0].min(binbounds[j][0]);
                    rb[1] = rb[1].min(binbounds[j][1]);
                    rb[2] = rb[2].min(binbounds[j][2]);
                    rb[3] = rb[3].max(binbounds[j][3]);
                    rb[4] = rb[4].max(binbounds[j][4]);
                    rb[5] = rb[5].max(binbounds[j][5]);
                }

                // get cost
                let cost = 3.0 + 1.0 + surface_area(lb) * ls as f32 + 1.0 + surface_area(rb) * rs as f32;
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = split;
                    best_aabb_left = lb;
                    best_aabb_right = rb;
                }
            }
        }

        if best_cost == max_cost { // leaf
            depth_timers[depth] += timer.elapsed().as_micros();
            handled += count;
            v.left_first = first; // first
            v.count = count;
            continue;
        }

        // partition
        let mut a = first; // first
        let mut b = first + count - 1; // last

        while a <= b {
            let bound = bounds[a];
            if ((bound[3+best_axis]-bound[best_axis])*0.5) < best_split{ // midpoint < best_split
                a += 1;
            } else {
                is.swap(a, b);
                // swap bounds a with b
                bounds.swap(a, b);
                b -= 1;
            }
        }
        let l_count = a - first;

        if l_count == 0 || l_count == count{ // leaf
            depth_timers[depth] += timer.elapsed().as_micros();
            handled += count;
            v.left_first = first; // first
            v.count = count;
            continue;
        }
        v.count = 0; // internal vertex, not a leaf
        v.left_first = poolptr; // left = poolptr, right = poolptr + 1
        poolptr += 2;
        let lf = v.left_first;

        depth_timers[depth] += timer.elapsed().as_micros();
        // todo prevent this stack push?
        stack.push(StackItem {current: lf,first,count: l_count, depth: depth + 1});
        stack.push(StackItem {current: lf+1,first: first+l_count,count: count-l_count, depth: depth + 1});
    }
    // println!("counter: {}" , counter);

    println!("depth_timers");
    let x = depth_timers.into_iter().map(|v| v / 1000).collect::<Vec<u128>>();
    let total_time : u128= x.iter().sum();
    println!("total_time: {:?}" , total_time);
    let mut depth_timer_sorted: Vec<(usize,u128)> = (0..x.len()).zip(x.into_iter()).collect();
    depth_timer_sorted.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    for item in depth_timer_sorted.into_iter() { println!("{:?}", item); }

    println!("depth_counters");
    let x = depth_counters;
    let mut depth_counter_sorted: Vec<(usize,u128)> = (0..x.len()).zip(x.into_iter()).collect();
    depth_counter_sorted.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    for item in depth_counter_sorted.into_iter() { println!("{:?}", item); }

    println!("depth_items");
    let x = depth_items;
    let total_items : usize = x.iter().sum();
    let mut depth_items_sorted: Vec<(usize,usize)> = (0..x.len()).zip(x.into_iter()).collect();
    depth_items_sorted.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    for item in depth_items_sorted.into_iter() { println!("{:?}", item); }
    println!("total_items: {:?}" , total_items);

}

const AABB_NULL : [f32;6] = [f32::MAX,f32::MAX,f32::MAX,f32::MIN,f32::MIN,f32::MIN];
#[inline]
fn union_bound(bounds: &[[f32;6]]) -> [f32;6] {
    let mut bound = AABB_NULL;
    for other in bounds{
        bound[0] = bound[0].min(other[0]) - EPSILON;
        bound[1] = bound[1].min(other[1]) - EPSILON;
        bound[2] = bound[2].min(other[2]) - EPSILON;
        bound[3] = bound[3].max(other[3]) + EPSILON;
        bound[4] = bound[4].max(other[4]) + EPSILON;
        bound[5] = bound[5].max(other[5]) + EPSILON;
    }
    bound
}

// #[inline]
// fn lerp(bound: [f32;6], ) -> {
//
// }

#[inline]
fn surface_area(bound: [f32;6]) -> f32 {
    let v = [bound[3] - bound[0], bound[4] - bound[1], bound[5] - bound[2]];
    v[0] * v[1] * 2.0 + v[0] * v[2] * 2.0 + v[1] * v[2] * 2.0
}
