use crate::scene::{Scene, Material, Triangle};
use crate::vec3::Vec3;

use obj::*;
use crate::bvh::{Bvh, Vertex};
use crate::aabb::AABB;

#[derive(Default)]
pub struct Mesh{
    pub name: String,
    pub triangles: Vec<Triangle>
}

impl Mesh {
    pub fn teapot() -> Self{
        Mesh::load_model("assets/models/teapot.obj")
    }
    pub fn dragon() -> Self{
        Mesh::load_model("assets/models/dragon.obj")
    }
    pub fn load_model(file_path: &str) -> Self{
        let mut mesh = Mesh {
            name: file_path.parse().unwrap(),
            triangles: vec![]
        };
        let obj = if let Ok(o) = Obj::load(file_path){ o }
        else { panic!("Could not load file: {}!", file_path); };
        let pos = obj.data.position;
        let mut tris = Vec::new();
        for ob in obj.data.objects{
            for group in ob.groups{
                for poly in group.polys{
                    // bullshit
                    if poly.0.len() < 3 { continue; }
                    // triangle
                    if poly.0.len() == 3{
                        let tri = [poly.0[0].0, poly.0[1].0, poly.0[2].0];
                        tris.push(tri);
                    }
                    else { // triangle fan
                        print!("fan, ");
                        for i in 0..poly.0.len() - 2{
                            let tri = [poly.0[i].0, poly.0[i + 1].0, poly.0[i + 2].0];
                            tris.push(tri);
                        }
                    }
                }
            }
        }
        // 5B = 5.000M = 50.000 * 100.000
        // we require 50.000 dragons
        // println!("model triangle size: {}", tris.len()); // dragon = 100.000
        for tri in &tris{
            let x0 = pos[tri[0]][0];
            let y0 = pos[tri[0]][1];
            let z0 = pos[tri[0]][2];
            let x1 = pos[tri[1]][0];
            let y1 = pos[tri[1]][1];
            let z1 = pos[tri[1]][2];
            let x2 = pos[tri[2]][0];
            let y2 = pos[tri[2]][1];
            let z2 = pos[tri[2]][2];
            mesh.triangles.push( Triangle{
                a: Vec3::new(x0, y0, z0),
                b: Vec3::new(x1, y1, z1),
                c: Vec3::new(x2, y2, z2),
            });
        }
        mesh
    }

    pub fn build_triangle_wall(mat: u8, scene: &mut Scene, diff: f32, offset: f32) -> Self{
        let mut mesh = Mesh {
            name: "triangle_wall".parse().unwrap(),
            triangles: vec![],
        };
        let z = 0.0;
        let mut x = -offset;
        let mut total = 0;
        while x < offset { // (2.0*offset)/diff
            let mut y = -offset;
            while y < offset { // 2.0*(2.0*offset)/diff
                let x0 = x;
                let y0 = y;
                let z0 = z;

                let x1 = x;
                let y1 = y + diff;
                let z1 = z;

                let x2 = x + diff;
                let y2 = y;
                let z2 = z;

                let x3 = x + diff;
                let y3 = y + diff;
                let z3 = z;
                mesh.triangles.push( Triangle{
                    a: Vec3::new(x0, y0, z0),
                    b: Vec3::new(x1, y1, z1),
                    c: Vec3::new(x2, y2, z2),
                });
                mesh.triangles.push( Triangle{
                    a: Vec3::new(x1, y1, z1),
                    b: Vec3::new(x2, y2, z2),
                    c: Vec3::new(x3, y3, z3),
                });
                total += 2;
                y += diff;
            }
            x += diff;
        }
        // println!("total:{}", total);
        mesh
    }
}

