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
    pub fn new_random() -> Self{
        Self {
            x: (random::<f32>() - 0.5) * MAX_RENDER_DIST,
            y: (random::<f32>() - 0.5) * MAX_RENDER_DIST,
            z: (random::<f32>() - 0.5) * MAX_RENDER_DIST
        }
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
#[derive(Default)]
pub struct Bvh{
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub mesh: Mesh
}

#[derive(Clone, Debug, Default)]
pub struct Vertex{
    pub bound: AABB,
    left_first: usize,
    count: usize,
}

pub struct BuilderData {
    bounds: Vec<AABB>,
    is: Vec<usize>,
    vs: Vec<Vertex>,
    bins: usize,
}

struct Info {
    maxdepth: usize,
    counter: usize,
    watch: Stopwatch,
    times: Vec<u128>,
}

impl Bvh{
    #[allow(clippy::too_many_arguments)]
    fn subdivide(data: &mut BuilderData){
        let current = 0;
        let first = 0;
        let mut poolptr = 2;
        let count = data.is.len();
        let mut depth = 0;

        let bounds = &data.bounds;
        let mut is = &mut data.is;
        let mut vs = &mut data.vs;
        let bins = data.bins;

        let binsf = bins as f32;
        let binsf_inf = 1.0 / binsf;

        // let mut stack = &mut CustomStack::new(); // [(current,first,count,depth,bound)]
        let mut stack = vec![];

        // let mut top_bound = Self::union_bound(is, bounds);
        stack.push(StackItem {current,first,count,depth});

        let mut lerps = vec![Vec3::ZERO; bins];
        let mut binbounds = vec![AABB::new();bins];
        let mut bincounts : Vec<usize> = vec![0;bins];
        let aabb_null = AABB::default();

        let (mut lb, mut rb) = (AABB::default(), AABB::default());

        let mut best_aabb_left = AABB::default();
        let mut best_aabb_right = AABB::default();
        let mut best_axis = Axis::X;
        let mut best_split = 0.0;

        let mut sub_is: &[usize];
        let mut v : &mut Vertex = &mut Vertex::default();

        let mut current = 0;
        let mut first = 0;
        let mut count = 0;
        let mut step = 0;

        let mut counter = 0;

        let mut timer = Stopwatch::start_new();
        let mut depth_timers = vec![];
        while stack.len() >= 0 {
            let mut x = stack.pop().unwrap();
            // measure time in depth
            depth = x.depth;
            if depth >= depth_timers.len() {
                depth_timers.push(0);
            }
            timer = Stopwatch::start_new();

            current = x.current;
            count = x.count;
            first = x.first;
            // println!("{:?}", x);
            v = &mut vs[current];
            sub_is = &is[first..first + count];

            let top_bound = &Self::union_bound(is, bounds);
            // let top_bound = &x.bound;
            v.bound.set(top_bound);

            if count < 3 { // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }
            // sah binned
            let diff = top_bound.max.subed(top_bound.min);
            let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

            // precompute lerps
            for (i, item) in lerps.iter_mut().enumerate(){
                item.x = top_bound.min.x + i as f32 * binsf_inf * diff.x;
                item.y = top_bound.min.y + i as f32 * binsf_inf * diff.y;
                item.z = top_bound.min.z + i as f32 * binsf_inf * diff.z;
            }

            // compute best combination; minimal cost
            let (mut ls, mut rs) = (0, 0);
            lb.set_default();
            rb.set_default();
            let mut best_cost = f32::MAX;

            for axis in [Axis::X, Axis::Y, Axis::Z] {

                let u = axis.as_usize();
                if !axis_valid[u] {
                    continue;
                }
                let k1 = (binsf*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
                let k0 = top_bound.min.fake_arr(axis);

                // place bounds in bins
                // generate bounds of bins
                binbounds.fill(AABB::ZERO);
                bincounts.fill(0);
                let mut index: usize ;
                for index_triangle in sub_is {
                    index = (k1*(data.bounds[*index_triangle].midpoint().as_array()[u]-k0)) as usize;
                    binbounds[index].combine(&data.bounds[*index_triangle]);
                    bincounts[index] += 1;
                }

                // iterate over bins
                for (lerp_index,lerp) in lerps.iter().enumerate(){
                    let split = lerp.fake_arr(axis);
                    // reset values
                    ls = 0;
                    rs = 0;
                    lb.set_default();
                    rb.set_default();
                    // construct bounds
                    for j in 0..lerp_index { // left of split
                        ls += bincounts[j];
                        lb.combine(&binbounds[j]);
                    }
                    for j in lerp_index..bins { // right of split
                        rs += bincounts[j];
                        rb.combine(&binbounds[j]);
                    }

                    // get cost
                    let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis;
                        best_split = split;
                        best_aabb_left.set(&lb);
                        best_aabb_right.set(&rb);
                    }
                }
            }

            // partition
            let mut a = first; // first
            let mut b = first + count - 1; // last
            let u = best_axis.as_usize();
            while a <= b{
                if data.bounds[is[a]].midpoint().as_array()[u] < best_split{
                    a += 1;
                } else {
                    is.swap(a, b);
                    b -= 1;
                }
            }
            let l_count = a - first;

            if l_count == 0 || l_count == count{ // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }
            v.count = 0; // internal vertex, not a leaf
            v.left_first = poolptr; // left = poolptr, right = poolptr + 1
            poolptr += 2;
            let lf = v.left_first;

            depth_timers[depth] += timer.elapsed().as_micros() as u128;

            stack.push(StackItem {current: lf,first,count: l_count, depth: depth + 1});
            stack.push(StackItem {current: lf+1,first: first+l_count,count: count-l_count, depth: depth + 1});
        }
        println!("counter: {}" , counter);
        let x = depth_timers.into_iter().map(|v| v as f64 * 0.001 as f64).collect::<Vec<f64>>();
        let total_time : f64= x.iter().sum();
        let mut tmp: Vec<(usize,f64)> = (0..x.len()).zip(x.into_iter()).collect();
        tmp.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        println!("depth_timers: {:?}" , tmp);
        // println!("depth_timers: {:?}" , tmp.sort_by(|a,b| (a.1).cmp(b.1)));
        // println!("depth_timers: {:?}" , tmp.sort_by_key(|a| a.1));
        // println!("depth_timers: {:?}" , tmp.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap()));
        println!("depth_timers: {:?}" , total_time);
        for item in tmp.into_iter() {
            println!("{:?}", item);
        }

    }

    fn union_bound(is: &[usize], bounds: &[AABB]) -> AABB {
        let mut bound = AABB::default();
        for i in is{
            bound.combine(&bounds[*i]);
        }
        bound.grown(Vec3::EPSILON)
    }

    pub fn get_prim_counts(&self, current: usize, vec: &mut Vec<usize>){
        if current >= self.vertices.len() { return; }
        let vs = &self.vertices;
        let v = &vs[current];
        if v.count > 0{ // leaf
            vec.push(v.count);
        } else { // vertex
            self.get_prim_counts(v.left_first, vec);
            self.get_prim_counts(v.left_first + 1, vec);
        }
    }

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){ (0,0) }
    pub fn from_mesh(mesh: Mesh, triangles: &Vec<Triangle>, bins: usize) -> Self{
        let n = triangles.len();
        if n == 0 {
            return Self{ indices: vec![], vertices: vec![], mesh };
        }
        let mut is = (0..n).into_iter().collect::<Vec<_>>();
        let mut vs = vec![Vertex::default(); n * 2];
        println!("Generating bounds...");
        let watch = Stopwatch::start_new();
        let bounds = (0..n).into_iter().map(|i|
            AABB::from_points(&[triangles[i].a, triangles[i].b, triangles[i].c])
        ).collect::<Vec<_>>();
        let mut elapsed = watch.elapsed_ms();
        println!("done building bounds in {}...", elapsed);
        println!("Generating midpoints...");
        let midpoints = (0..n).into_iter().map(|i|
            bounds[i].midpoint().as_array()
        ).collect::<Vec<_>>();
        let mut elapsed = watch.elapsed_ms();
        println!("done building midpoints in {}...", elapsed);
        let mut data = BuilderData {
            bounds,
            is,
            vs,
            bins,
        };
        let watch = Stopwatch::start_new();
        Self::subdivide(&mut data);
        println!("{:?}", watch.elapsed_ms());

        let p = data.is.into_iter().map(|i| i as u32).collect();
        Self{
            indices: p,
            vertices: data.vs,
            mesh
        }
    }
}


const STACK_SIZE : usize = 100000;
pub struct CustomStack {
    pub stack: [StackItem; STACK_SIZE],
    pub index: usize
}

impl CustomStack {
    pub fn new() -> Self{
        const INIT: StackItem = StackItem {
            current: 0,
            first: 0,
            count: 0,
            depth: 0,
            // bound: AABB::ZERO
        };
        Self { stack: [INIT; STACK_SIZE], index: 0 }
    }
    #[inline]
    pub fn current(&self) -> &StackItem{
        &self.stack[self.index]
    }
    pub fn push(&mut self, item: StackItem){
        self.index += 1;
        self.stack[self.index] = item;
    }
    pub fn pop(&mut self) -> &StackItem{
        self.index = self.index-1;
        &self.stack[self.index + 1]
    }
}

#[derive(Default, Clone, Debug)]
pub struct StackItem {
    pub current : usize,
    pub first : usize,
    pub count : usize,
    pub depth : usize,
    // pub bound : AABB
}

// credit: George Marsaglia
#[inline]
fn xor32(seed: &mut u32) -> u32{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

const TRIANGLES : usize = 50000;

fn main() {
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

}