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
    pub fn as_array(&self) -> [f32;3]{
        [self.x,self.y,self.z]
    }

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
        v.x * v.y * 2.0 +
        v.x * v.z * 2.0 +
        v.y * v.z * 2.0
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

// main
#[inline]
fn xor32(seed: &mut u32) -> u32{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

// type Vectorf32 = Simd<f32, 32>;
// type Vectorusize = Simd<usize, 32>;
// struct SimdCacher {
//     pub idxs: Vec<Vectorusize>
// }
// impl SimdCacher {
//     fn init() -> Self {
//         // const reshuffle indices
//         let mut idxs = vec![0;32]; // todo: use arrayvec
//         let mut idxs_slices: Vec<Vectorusize> = vec![];  // todo: use arrayvec
//         for i in 0..6{
//             for j in 0..32 {
//                 idxs[j] = j*6+i;
//             }
//             idxs_slices.push(Vectorusize::from_slice(idxs.as_slice()));
//         }
//         Self {
//             idxs: idxs_slices
//         }
//     }
// }
// fn min_box_aabbs() {
//     // generate bounding boxes
//     let mut aabbs = vec![];
//     let mut seed:u32 = 81349324; // guaranteed to be random
//     let timer = Stopwatch::start_new();
//     for i in 0..50000000{
//         aabbs.push(AABB {
//             min: Vec3 {
//                 x: xor32(&mut seed) as f32,
//                 y: xor32(&mut seed) as f32,
//                 z: xor32(&mut seed) as f32
//             },
//             max: Vec3 {
//                 x: xor32(&mut seed) as f32,
//                 y: xor32(&mut seed) as f32,
//                 z: xor32(&mut seed) as f32
//             }
//         });
//     }
//     println!("generating bounding boxes:  {:?}", timer.elapsed_ms());
//     // combine  box as usual
//     let timer = Stopwatch::start_new();
//     let mut combox = AABB::default();
//     for i in &aabbs {
//         combox.combine(*i);
//     }
//     println!("normal {:?}", timer.elapsed_ms());
//     println!("combox {:?}", combox);
//
//     // convert aabbs to large vec
//     let mut combox: Vec<f32> = aabbs.into_iter().map(|v| vec![v.min.x, v.min.y, v.min.z, v.max.x, v.max.y, v.max.z]).flatten().collect();
//     let mut index = 1;
//     let combox_len = combox.len();
//     // while index + 35*6 < combox_len { combox[0] = }
//     // while combox.len() > 32*6 { combox = combox.chunks_exact(32*6).map(|aabbs| aab_min_32(aabbs)).collect(); }
//     // while combox.len() > 6 { combox = combox.combine(*i); }
//     let cacher = SimdCacher::init();
//     let aabb_chunk_size = 31*6;
//     let timer = Stopwatch::start_new();
//     while index + aabb_chunk_size < combox_len {
//         for i in 0..6 {
//             // combox[i] = combox.chunks_exact(32*6).map(|aabbs| aab_min_32(aabbs)).collect();
//             combox[i] = Vectorf32::gather_or_default(&combox, cacher.idxs[i]).horizontal_min();
//         }
//         index += aabb_chunk_size;
//     }
//     println!("simd {:?}", timer.elapsed_ms());
//     let combox = AABB {
//         min: Vec3 {
//             x: combox[0],
//             y: combox[1],
//             z: combox[2],
//         },
//         max: Vec3 {
//             x: combox[3],
//             y: combox[4],
//             z: combox[5],
//         },
//     };
//     println!("combox {:?}", combox);
// }

fn generate_aabbs() -> Vec<AABB> {
    let mut aabbs = vec![];
    let mut seed :u32 = 81349324; // guaranteed to be random
    let first = seed;
    // for i in 0..50000000{
        // assert_ne!(xor32(&mut seed), first);
    // }
    for i in 0..50000000{
        aabbs.push(AABB {
            min: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
            max: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            }
        });
    }
    aabbs
}

fn bench_combine(aabbs: &[AABB]) {
    let mut combox = AABB::default();
    for i in aabbs {
        combox.combine(*i);
    }
}

fn bench_lerp(aabbs: &[AABB]) {
    let bins = 12;
    let binsf = bins as f32;
    let binsf_inf = 1.0 / binsf;
    let mut lerps = vec![Vec3::ZERO; bins];
    for i in 0..100 {
        for bound in aabbs {
            for (i, item) in lerps.iter_mut().enumerate(){
                *item = bound.lerp(i as f32 * binsf_inf);
            }
        }
    }
}

fn bench_place_bins(aabbs: &[AABB]) {
    // let bins = 12;
    // let binsf = bins as f32;
    // let binsf_inf = 1.0 / binsf;
    // let mut binbounds = vec![AABB::new();bins];
    // let mut bincounts : Vec<usize> = vec![0;bins];
    // binbounds.fill(aabb_null);
    // bincounts.fill(0);
    // let mut index: usize ;
    // for index_triangle in sub_range.clone() {
    //     index = (k1*(bounds[index_triangle].midpoint().as_array()[u]-k0)) as usize;
    //     binbounds[index].combine(bounds[index_triangle]);
    //     bincounts[index] += 1;
    // }
}

fn benchmarker(aabbs: &[AABB], call: Box<Fn(&[AABB]) -> ()>) {
    let timer = Stopwatch::start_new();
    call(aabbs);
    println!("bench: {}", timer.elapsed_ms());
}

fn main() {
    println!("start");
    let aabbs = generate_aabbs();
    let mut f32s: Vec<f32> = aabbs.iter().map(|v| vec![v.min.x, v.min.y, v.min.z, v.max.x, v.max.y, v.max.z]).flatten().collect();

    benchmarker(aabbs.as_slice(), Box::new(bench_combine));
    benchmarker(aabbs.as_slice(), Box::new(bench_lerp));
    benchmarker(aabbs.as_slice(), Box::new(bench_place_bins));

    // bench_combine(aabbs.as_slice());
    // bench_lerp(aabbs.as_slice());
    // println!("{}", f32s.len());
    // let mut combox = AABB::default();
    // for i in &aabbs {
    //     combox.combine(*i);
    // }
    // println!("{:?}", combox);
    // println!("{:?}", &aabbs[0..10]);


}
