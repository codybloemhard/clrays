extern crate clrays_rs;
use clrays_rs as clr;
use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,Material,Plane,Sphere,Light};
use clr::vec3::Vec3;
use clr::kernels::{VoidKernel,ResultKernel,TraceKernel};
use clr::cl_helpers::{create_five,ClBuffer};

pub fn main() -> Result<(),String>{
    //clr::test(PlatformTest::OpenCl2);
    let mut scene = Scene::new();
    scene.sky_col = Vec3::new(0.2, 0.2, 0.9).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::up();
    scene.cam_dir = Vec3::backward();
    //scene.add_texture("sky".to_string(), "../Assets/Textures/sky1.jpg".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    //scene.add_texture("wood".to_string(), "../Assets/Textures/wood.png".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    //scene.set_skybox("wood");
    scene.add_plane(Plane{
        pos: Vec3::zero(),
        nor: Vec3::up(),
        mat: Material::basic().with_colour(Vec3::new(0.1, 1.0, 0.1)),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(-1.0, 1.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_colour(Vec3::new(1.0, 0.1, 0.1))
            .with_roughness(1.0),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(1.0, 1.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_colour(Vec3::new(0.1,0.1,1.0))
            .with_roughness(0.5)
            .with_reflectivity(0.5),
    });
    scene.add_light(Light{
        pos: Vec3::up(),
        intensity: 50.0,
        col: Vec3::one(),
    });

    use std::io::prelude::*;
    let mut file = std::fs::File::open("../Assets/Kernels/raytrace.cl").unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).expect("file to string aaah broken");

    let (_,_,_,program,queue) = create_five(&src).expect("expect: create big five");

    let w = 960u32;
    let h = 540u32;
    let mut buffer = ClBuffer::<i32>::new(&queue, w as usize * h as usize, 0)
        .expect("expect test: make buffer");
    let scene_raw = scene.get_buffers();
    let scene_params_raw = scene.get_params_buffer();
    let scene_params = ClBuffer::from(&queue, scene_params_raw)
        .expect("expect test: scene params buffer");
    let scene_items = ClBuffer::from(&queue, scene_raw)
        .expect("expect test: scene items buffer");
    let tex_raw = scene.get_textures_buffer();
    let tex_params_raw = scene.get_texture_params_buffer();
    let tex_params = ClBuffer::from(&queue, tex_params_raw)
        .expect("expect test: tex params buffer");
    let tex_items = ClBuffer::from(&queue, tex_raw)
        .expect("expect test: tex items buffer");
    let kernel = ocl::Kernel::builder()
    .program(&program)
    .name("raytracing")
    .queue(queue.clone())
    .global_work_size([w,h])
    .arg(buffer.get_ocl_buffer())
    .arg(w as u32)
    .arg(h as u32)
    .arg(scene_params.get_ocl_buffer())
    .arg(scene_items.get_ocl_buffer())
    .arg(tex_params.get_ocl_buffer())
    .arg(tex_items.get_ocl_buffer())
    .build().expect("expect test: build kernel");

    unsafe {
        kernel.cmd().queue(&queue).enq().expect("expect test: run kernel");
    }

    buffer.download(&queue).expect("expect test: download buffer");
    let tex = buffer.get_slice();

    let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    window.run(window::std_input_handler, Some(tex));
    Ok(())
}
