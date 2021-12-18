use core::simd::*;
use arrayvec::ArrayVec;
use rand::random;

pub fn myfunc() {
    let mut arr: [i32; 128] = [0; 128];
    for (elem, val) in arr.iter_mut().zip(1..=128) {
        *elem = val;
    }
    // sum

    let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    let idxs : Simd<i32,4>= Simd::from_array([9, 3, 0, 0]);
    // let vals = from_array([-27, 82, -41, 124]);

    let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    let idxs = Simd::from_array([9, 3, 0, 5]);
    let alt = Simd::from_array([-5, -4, -3, -2]);

    let result = Simd::gather_or(&vec, idxs, alt); // Note the lane that is out-of-bounds.
    assert_eq!(result, Simd::from_array([-5, 13, 10, 15]));


    // let lots_of_3s = (&[-123.456f32; 128][..])
    //     .simd_iter()
    //     .simd_map(f32s(0.0), |v| {
    //         f32s(9.0) * v.abs().sqrt().rsqrt().ceil().sqrt() - f32s(4.0) - f32s(2.0)
    //     })
    //     .scalar_collect();
    panic!();
}
pub fn my_simd_sum() {
    const LANES : usize = 64;
    type Scalar = i32;
    type Vector<const LANES: usize> = Simd<Scalar, LANES>;
    let mut arr: [i32; LANES] = [0; LANES];
    for (elem, val) in arr.iter_mut().zip(1..=128) { *elem = val; }
    println!("{:?}",arr);
    let vmax = Vector::<LANES>::from_array(arr).horizontal_max();
    // let simd_arr = Simd::from_array(arr,    );
    // println!("{:?}",simd_arr.min());
    println!("{:?}",vmax);
}


pub fn iterator_stuff() {
    const LANES : usize = 8;
    type Scalar = i32;
    type Vector<const LANES: usize> = Simd<Scalar, LANES>;
    type Vector4 = Simd<Scalar, 4>;
    type Vector8 = Simd<Scalar, 8>;
    type Vector16 = Simd<Scalar, 16>;
    // #[derive(Default, Debug, Copy, Clone)]
    type AAB = [Scalar;6];
    // generate a few hundred bounds
    let mut aabs: [AAB; 400] = [[0,0,0,0,0,0]; 400];
    for (i, aab) in aabs.iter_mut().enumerate() {
        aab[0] = 8021321* random::<Scalar>();
        aab[1] = 021321* random::<Scalar>();
        aab[2] = 801321* random::<Scalar>();
        aab[3] = 80131* random::<Scalar>();
        aab[4] = 8021* random::<Scalar>();
        aab[5] = 801321* random::<Scalar>();
    }
    println!("{:?}", aabs);
    // map to first element

    fn aab_min_4(bbs: &[AAB]) -> AAB {
        [
            Vector4::from_array([bbs[0][0], bbs[1][0], bbs[2][0], bbs[3][0]]).horizontal_min(),
            Vector4::from_array([bbs[0][1], bbs[1][1], bbs[2][1], bbs[3][1]]).horizontal_min(),
            Vector4::from_array([bbs[0][2], bbs[1][2], bbs[2][2], bbs[3][2]]).horizontal_min(),
            Vector4::from_array([bbs[0][3], bbs[1][3], bbs[2][3], bbs[3][3]]).horizontal_min(),
            Vector4::from_array([bbs[0][4], bbs[1][4], bbs[2][4], bbs[3][4]]).horizontal_min(),
            Vector4::from_array([bbs[0][5], bbs[1][5], bbs[2][5], bbs[3][5]]).horizontal_min(),
        ]
    }

    fn aab_min_8(bbs: &[AAB]) -> AAB {
        [
            Vector8::from_array([bbs[0][0],bbs[1][0],bbs[2][0],bbs[3][0],bbs[4][0],bbs[5][0],bbs[6][0],bbs[7][0]]).horizontal_min(),
            Vector8::from_array([bbs[0][1],bbs[1][1],bbs[2][1],bbs[3][1],bbs[4][1],bbs[5][1],bbs[6][1],bbs[7][1]]).horizontal_min(),
            Vector8::from_array([bbs[0][2],bbs[1][2],bbs[2][2],bbs[3][2],bbs[4][2],bbs[5][2],bbs[6][2],bbs[7][2]]).horizontal_min(),
            Vector8::from_array([bbs[0][3],bbs[1][3],bbs[2][3],bbs[3][3],bbs[4][3],bbs[5][3],bbs[6][3],bbs[7][3]]).horizontal_min(),
            Vector8::from_array([bbs[0][4],bbs[1][4],bbs[2][4],bbs[3][4],bbs[4][4],bbs[5][4],bbs[6][4],bbs[7][4]]).horizontal_min(),
            Vector8::from_array([bbs[0][5],bbs[1][5],bbs[2][5],bbs[3][5],bbs[4][5],bbs[5][5],bbs[6][5],bbs[7][5]]).horizontal_min(),
        ]
    }

    fn aab_min_16(bbs: &[AAB]) -> AAB {
        [
            Vector16::from_array([bbs[0][0],bbs[1][0],bbs[2][0],bbs[3][0],bbs[4][0],bbs[5][0],bbs[6][0],bbs[7][0],bbs[8][0],bbs[9][0],bbs[10][0],bbs[11][0],bbs[12][0],bbs[13][0],bbs[14][0],bbs[15][0]]).horizontal_min(),
            Vector16::from_array([bbs[0][1],bbs[1][1],bbs[2][1],bbs[3][1],bbs[4][1],bbs[5][1],bbs[6][1],bbs[7][1],bbs[8][1],bbs[9][1],bbs[10][1],bbs[11][1],bbs[12][1],bbs[13][1],bbs[14][1],bbs[15][1]]).horizontal_min(),
            Vector16::from_array([bbs[0][2],bbs[1][2],bbs[2][2],bbs[3][2],bbs[4][2],bbs[5][2],bbs[6][2],bbs[7][2],bbs[8][2],bbs[9][2],bbs[10][2],bbs[11][2],bbs[12][2],bbs[13][2],bbs[14][2],bbs[15][2]]).horizontal_min(),
            Vector16::from_array([bbs[0][3],bbs[1][3],bbs[2][3],bbs[3][3],bbs[4][3],bbs[5][3],bbs[6][3],bbs[7][3],bbs[8][3],bbs[9][3],bbs[10][3],bbs[11][3],bbs[12][3],bbs[13][3],bbs[14][3],bbs[15][3]]).horizontal_min(),
            Vector16::from_array([bbs[0][4],bbs[1][4],bbs[2][4],bbs[3][4],bbs[4][4],bbs[5][4],bbs[6][4],bbs[7][4],bbs[8][4],bbs[9][4],bbs[10][4],bbs[11][4],bbs[12][4],bbs[13][4],bbs[14][4],bbs[15][4]]).horizontal_min(),
            Vector16::from_array([bbs[0][5],bbs[1][5],bbs[2][5],bbs[3][5],bbs[4][5],bbs[5][5],bbs[6][5],bbs[7][5],bbs[8][5],bbs[9][5],bbs[10][5],bbs[11][5],bbs[12][5],bbs[13][5],bbs[14][5],bbs[15][5]]).horizontal_min(),
        ]
    }

    println!("{:?}",aab_min_4(&aabs[0..4]));
    println!("{:?}",aab_min_8(&aabs[0..8]));
    println!("{:?}",aab_min_16(&aabs[0..16]));

    // iterate in ranges of 8
}

// #[inline]
// fn min32(bbs:&[f32]) -> f32 {
//     type Vector32 = Simd<f32, 32>;
//     Vector32::from_array([bbs[0],bbs[1],bbs[2],bbs[3],bbs[4],bbs[5],bbs[6],bbs[7],bbs[8],bbs[9],bbs[10],bbs[11],bbs[12],bbs[13],bbs[14],bbs[15],bbs[16],bbs[17],bbs[18],bbs[19],bbs[20],bbs[21],bbs[22],bbs[23],bbs[24],bbs[25],bbs[26],bbs[27],bbs[28],bbs[29],bbs[30],bbs[31]]).horizontal_min()
// }
//
// #[inline] // todo use arrayvec
// fn aab_min_32(bbs: &[f32]) -> Vec<f32> { // (32 * 6) / 32 = 6
//     let mut result = vec![0.0;6];
//     for i in 0..6{ result[i] = Vectorf32::gather_or_default(bbs, idxs_slices[i]).horizontal_min(); }
//     result
// }

type Vectorf32 = Simd<f32, 32>;
type Vectorusize = Simd<usize, 32>;
struct SimdCacher {
    pub idxs: Vec<Vectorusize>
}
impl SimdCacher {
    fn init() -> Self {
        // const reshuffle indices
        let mut idxs = vec![0;32]; // todo: use arrayvec
        let mut idxs_slices: Vec<Vectorusize> = vec![];  // todo: use arrayvec
        for i in 0..6{
            for j in 0..32 {
                idxs[j] = j*6+i;
            }
            idxs_slices.push(Vectorusize::from_slice(idxs.as_slice()));
        }
        Self {
            idxs: idxs_slices
        }
    }
}

fn min_box_aabbs() {
    // generate bounding boxes
    let mut aabbs = vec![];
    let timer = Stopwatch::start_new();
    for i in 0..50000000{
        aabbs.push(AABB {
            min: Vec3 {
                x: random::<f32>(),
                y: random::<f32>(),
                z: random::<f32>(),
            },
            max: Vec3 {
                x: random::<f32>(),
                y: random::<f32>(),
                z: random::<f32>(),
            }
        });
    }
    println!("generating bounding boxes:  {:?}", timer.elapsed_ms());
    // combine  box as usual
    let timer = Stopwatch::start_new();
    let mut combox = AABB::default();
    for i in &aabbs {
        combox.combine(*i);
    }
    println!("normal {:?}", timer.elapsed_ms());
    println!("combox {:?}", combox);

    // convert aabbs to large vec
    let mut combox: Vec<f32> = aabbs.into_iter().map(|v| vec![v.min.x, v.min.y, v.min.z, v.max.x, v.max.y, v.max.z]).flatten().collect();
    let mut index = 1;
    let combox_len = combox.len();
    // while index + 35*6 < combox_len { combox[0] = }
    // while combox.len() > 32*6 { combox = combox.chunks_exact(32*6).map(|aabbs| aab_min_32(aabbs)).collect(); }
    // while combox.len() > 6 { combox = combox.combine(*i); }
    let cacher = SimdCacher::init();
    let aabb_chunk_size = 31*6;
    let timer = Stopwatch::start_new();
    while index + aabb_chunk_size < combox_len {
        for i in 0..6 {
            // combox[i] = combox.chunks_exact(32*6).map(|aabbs| aab_min_32(aabbs)).collect();
            combox[i] = Vectorf32::gather_or_default(&combox, cacher.idxs[i]).horizontal_min();
        }
        index += aabb_chunk_size;
    }
    println!("simd {:?}", timer.elapsed_ms());
    let combox = AABB {
        min: Vec3 {
            x: combox[0],
            y: combox[1],
            z: combox[2],
        },
        max: Vec3 {
            x: combox[3],
            y: combox[4],
            z: combox[5],
        },
    };
    println!("combox {:?}", combox);
}


fn quicksorter() {

}



// type AAB = [Scalar;6];
const LANES : usize = 8;
type Scalar = i32;
struct AAB([Scalar;6]);
type Vector<const LANES: usize> = Simd<Scalar, LANES>;
type Vector4 = Simd<Scalar, 4>;
type Vector8 = Simd<Scalar, 8>;
type Vector16 = Simd<Scalar, 16>;
// impl FromIterator<Scalar> for AAB {
//     fn from_iter<I: IntoIterator<Item=i32>>(iter: I) -> Self{
//         AAB [
//             iter.next().unwrap(),
//             iter.next().unwrap(),
//             iter.next().unwrap(),
//             iter.next().unwrap(),
//             iter.next().unwrap(),
//             iter.next().unwrap(),
//         ]
//     }
// }

pub fn iterator_aabbs() {
    // #[derive(Default, Debug, Copy, Clone)]
    // generate a few hundred bounds
    // let x = AAB([0,0,0,0,0,0]);
    // let mut aabs: Vec<AAB> = vec![AAB[0,0,0,0,0,0];65536];
    // for (i, aab) in aabs.iter_mut().enumerate() {
    //     aab[0] = random::<Scalar>();
    //     aab[1] = random::<Scalar>();
    //     aab[2] = random::<Scalar>();
    //     aab[3] = random::<Scalar>();
    //     aab[4] = random::<Scalar>();
    //     aab[5] = random::<Scalar>();
    // }
    // fn aab_min_16(bbs: &[AAB]) -> AAB {
    //     [
    //         Vector16::from_array([bbs[0][0],bbs[1][0],bbs[2][0],bbs[3][0],bbs[4][0],bbs[5][0],bbs[6][0],bbs[7][0],bbs[8][0],bbs[9][0],bbs[10][0],bbs[11][0],bbs[12][0],bbs[13][0],bbs[14][0],bbs[15][0]]).horizontal_min(),
    //         Vector16::from_array([bbs[0][1],bbs[1][1],bbs[2][1],bbs[3][1],bbs[4][1],bbs[5][1],bbs[6][1],bbs[7][1],bbs[8][1],bbs[9][1],bbs[10][1],bbs[11][1],bbs[12][1],bbs[13][1],bbs[14][1],bbs[15][1]]).horizontal_min(),
    //         Vector16::from_array([bbs[0][2],bbs[1][2],bbs[2][2],bbs[3][2],bbs[4][2],bbs[5][2],bbs[6][2],bbs[7][2],bbs[8][2],bbs[9][2],bbs[10][2],bbs[11][2],bbs[12][2],bbs[13][2],bbs[14][2],bbs[15][2]]).horizontal_min(),
    //         Vector16::from_array([bbs[0][3],bbs[1][3],bbs[2][3],bbs[3][3],bbs[4][3],bbs[5][3],bbs[6][3],bbs[7][3],bbs[8][3],bbs[9][3],bbs[10][3],bbs[11][3],bbs[12][3],bbs[13][3],bbs[14][3],bbs[15][3]]).horizontal_min(),
    //         Vector16::from_array([bbs[0][4],bbs[1][4],bbs[2][4],bbs[3][4],bbs[4][4],bbs[5][4],bbs[6][4],bbs[7][4],bbs[8][4],bbs[9][4],bbs[10][4],bbs[11][4],bbs[12][4],bbs[13][4],bbs[14][4],bbs[15][4]]).horizontal_min(),
    //         Vector16::from_array([bbs[0][5],bbs[1][5],bbs[2][5],bbs[3][5],bbs[4][5],bbs[5][5],bbs[6][5],bbs[7][5],bbs[8][5],bbs[9][5],bbs[10][5],bbs[11][5],bbs[12][5],bbs[13][5],bbs[14][5],bbs[15][5]]).horizontal_min(),
    //     ]
    // }
    // make chunks of 16
    // while aabs.len() > 1 {
    //     let mut res = Vec::new();
    //     res.reserve(aabs.len()/16);
    //     aabs.chunks_exact(16).into_iter().map(|aabs| aab_min_16(aabs));
    //
    //     let t  = aabs.chunks_exact(16).into_iter().map(|aabs| aab_min_16(aabs)).flatten().collect();;
    //     aabs = t;
    // }
    // println!("{:?}",aabs);

}

// pub fn simd_aabb_min() {
//     const LANES : usize = 8;
//     type Scalar = i32;
//     type Vector<const LANES: usize> = Simd<Scalar, LANES>;
//     // #[derive(Default, Debug, Copy, Clone)]
//     type AAB = [Scalar;6];
//
//     // generate a few hundred bounds
//     let mut aabs: [AAB; 400] = [AAB::default(); 400];
//     let res : Vec<AAB> = aabs.chunks_exact(8)
//         .map(|vec| [
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[0]).collect()).horizontal_min(),
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[1]).collect()).horizontal_min(),
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[2]).collect()).horizontal_min(),
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[3]).collect()).horizontal_min(),
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[4]).collect()).horizontal_min(),
//             Vector::<LANES>::from_array(vec.iter().map(|aab| aab[5]).collect()).horizontal_min(),
//         ]).collect();
//             // a: [Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.a[0]).collect()).horizontal_min(),
//             //     Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.a[1]).collect()).horizontal_min(),
//             //     Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.a[2]).collect()).horizontal_min()],
//             // b: [Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.b[0]).collect()).horizontal_max(),
//             //     Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.b[1]).collect()).horizontal_max(),
//             //     Vector::<LANES>::from_array(vec.into_iter().map(|aab| aab.b[2]).collect()).horizontal_max()],
//             // a: [Vector::<LANES>::from_array(vec.iter().map(|aab| *aab.a[0]).collect()).horizontal_min(),
//         //     a: [Vector::<LANES>::from_array(vec).horizontal_min(),
//         //         Vector::<LANES>::from_array([0,1,2,3,4,5,6,7]).horizontal_min(),
//         //         Vector::<LANES>::from_array([0,1,2,3,4,5,6,7]).horizontal_min()],
//         //     b: [Vector::<LANES>::from_array([0,1,2,3,4,5,6,7]).horizontal_max(),
//         //         Vector::<LANES>::from_array([0,1,2,3,4,5,6,7]).horizontal_max(),
//         //         Vector::<LANES>::from_array([0,1,2,3,4,5,6,7]).horizontal_max()],
//         // }).collect();
//     assert_eq!(res.len(), 50);
//     println!("{:?}",res);
// }

pub fn my_simd_min() {
    const LANES : usize = 8;
    type Scalar = i32;
    type Vector<const LANES: usize> = Simd<Scalar, LANES>;

    let mut arr: [i32; LANES] = [0; LANES];
    for (elem, val) in arr.iter_mut().zip(1..=128) { *elem = val; }
    let vmax = Vector::<LANES>::from_array(arr);
}


use crate::scene::{Scene, Either, Model, ModelIndex, MeshIndex, Triangle};
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::mesh::Mesh;
use std::sync::Arc;
use stopwatch::Stopwatch;
use std::iter::FromIterator;


// use simdeez::*;
// use simdeez::scalar::*;
// use simdeez::sse2::*;
// use simdeez::sse41::*;
// use simdeez::avx::*;
// use simdeez::avx2::*;

// simd_runtime_generate!(
//     pub fn simd_width() {
//         println!("{}", S::VF32_WIDTH);
//     }
// );
// simd_compiletime_generate!(
//     pub fn simd_width() {
//         println!("{}", S::VF32_WIDTH);
//     }
//     // fn max(
//     //     x1: &[f32],
//     //     y1: &[f32],
//     //     x2: &[f32],
//     //     y2: &[f32]) -> f32 {
//     //
//     //
//     //     0.0
//     // }
// );

#[derive(Default)]
pub struct Bvh{
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub mesh: Mesh
}

#[derive(Clone, Copy, Debug, Default)]
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
    info: Info,
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

        let mut stack = &mut CustomStack::<StackItem>::new(); // [(current,first,count,step)]
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
        let mut top_bound;

        let mut current = 0;
        let mut first = 0;
        let mut count = 0;
        let mut step = 0;

        let mut counter = 0;

        let mut timer = Stopwatch::start_new();
        let mut depth_timers = vec![];
        while stack.index >= 1 {
            let mut x =  stack.pop();
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
            top_bound = Self::union_bound(sub_is, bounds);
            v.bound = top_bound;

            if count < 3 { // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }
            // sah binned
            let diff = top_bound.max.subed(top_bound.min);
            let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

            // precompute lerps
            // lerps.fill(Vec3::ZERO);
            for (i, item) in lerps.iter_mut().enumerate(){
                *item = top_bound.lerp(i as f32     * binsf_inf);
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
                binbounds.fill(aabb_null);
                bincounts.fill(0);
                let mut index: usize ;
                for index_triangle in sub_is {
                    index = (k1*(data.bounds[*index_triangle].midpoint().as_array()[u]-k0)) as usize;
                    binbounds[index].combine(data.bounds[*index_triangle]);
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
                        lb.combine(binbounds[j]);
                    }
                    for j in lerp_index..bins { // right of split
                        rs += bincounts[j];
                        rb.combine(binbounds[j]);
                    }

                    // get cost
                    let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis;
                        best_split = split;
                        best_aabb_left = lb;
                        best_aabb_right = rb;
                    }
                }
            }

            // partition
            data.info.watch.start();
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
            bound.combine(bounds[*i]);
        }
        bound.grown(Vec3::EPSILON)
    }

    pub fn get_prim_counts(&self, current: usize, vec: &mut Vec<usize>){
        if current >= self.vertices.len() { return; }
        let vs = &self.vertices;
        let v = vs[current];
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
            info: Info {
                maxdepth: 0,
                counter: 0,
                watch,
                times: vec![0,0,0,0,0],
            }
        };
        let watch = Stopwatch::start_new();
        Self::subdivide(&mut data);
        println!("{:?}", data.info.times.iter().map(|t| *t as f64 * 0.000001).collect::<Vec<f64>>());
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

#[cfg(test)]
mod test {
    use crate::bvh;
    use core::simd::*;
    use crate::bvh::{iterator_stuff, min_box_aabbs};

    #[test]
    fn first_test() {
        let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
        let idxs = Simd::from_array([9, 3, 0, 5]);
        let alt = Simd::from_array([-5, -4, -3, -2]);
        let result = Simd::gather_or(&vec, idxs, alt); // Note the lane that is out-of-bounds.
        assert_eq!(result, Simd::from_array([-5, 13, 10, 15]));
    }

    #[test]
    fn simd_sum() {
        bvh::my_simd_sum();
        panic!();
    }

    #[test]
    fn test_simd_aab_min() {
        iterator_stuff();
        // simd_aabb_min();
        panic!();
    }

    #[test]
    fn unit_aabb_min() {
        min_box_aabbs();
        panic!();
    }

}
