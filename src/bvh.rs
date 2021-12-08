use crate::scene::{ Scene, AABB };

#[derive(Clone, PartialEq, Debug)]
pub struct Bvh{
    prim_indices: Vec<u32>,
    vertices: Vec<Vertex>,
}

impl Bvh{
    pub fn from(scene: &Scene) -> Self{
        let prims = scene.spheres.len() + scene.triangles.len();

        let mut is = (0..prims as u32).into_iter().collect::<Vec<_>>();
        let mut vs = vec![Vertex::default(); prims * 2 - 1];
        let mut poolptr = 2;

        let mut outer_bound = scene.spheres.iter().fold(AABB::default(), |acc, x| acc.combined(AABB::from_point_radius(x.pos, x.rad)));
        outer_bound = scene.triangles.iter().fold(outer_bound, |acc, x| acc.combined(AABB::from_points(&[x.a, x.b, x.c])));

        let root = &mut vs[0];
        root.bound = outer_bound;
        root.left_first = 0;
        root.count = prims as u32;

        Self::subdivide(scene, &mut is, &mut vs, 0, &mut poolptr);

        Self{
            prim_indices: is,
            vertices: vs,
        }
    }

    fn subdivide(scene: &Scene, is: &mut[u32], vs: &mut[Vertex], current: usize, poolptr: &mut u32){
        let mut v = vs[current];
        if v.count < 3 { return; }
        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;
        Self::partition();
        Self::subdivide(scene, is, vs, v.left_first as usize, poolptr);
        Self::subdivide(scene, is, vs, v.left_first as usize + 1, poolptr);
    }

    fn partition(){

    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Vertex{
    bound: AABB,
    left_first: u32,
    count: u32,
}
