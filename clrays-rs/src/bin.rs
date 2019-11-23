#[macro_use] 
extern crate clrays_rs;
use clrays_rs as clr;
use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,Material,Plane,Sphere,Light};
use clr::vec3::{Vec3,BasicColour};
use clr::kernels::{VoidKernel,ResultKernel,TraceKernel};
use clr::cl_helpers::{create_five};
use clr::misc::{load_source};
use clr::info::{Info};

pub fn main() -> Result<(),String>{
    //clr::test(PlatformTest::OpenCl2);
    let mut info = Info::new();
    let mut scene = Scene::new();
    scene.sky_col = Vec3::soft_colour(BasicColour::Blue, 0.9, 0.2).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::up();
    scene.cam_dir = Vec3::backward();
    scene.add_texture("sky".to_string(), "../Assets/Textures/sky1.jpg".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    scene.add_texture("wood".to_string(), "../Assets/Textures/wood.png".to_string(), clr::trace_tex::TexType::Vector3c8bpc);
    scene.set_skybox("sky", &mut info);
    
    let woodtex = scene.get_texture("wood".to_string(), &mut info);
    scene.add_plane(Plane{
        pos: Vec3::zero(),
        nor: Vec3::up(),
        mat: Material::basic()
            .with_texture(woodtex),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(-1.0, 1.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_colour(Vec3::std_colour(BasicColour::Red))
            .with_roughness(1.0),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(1.0, 1.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_colour(Vec3::std_colour(BasicColour::Blue))
            .with_roughness(0.1)
            .with_reflectivity(0.5),
    });
    scene.add_light(Light{
        pos: Vec3::new(1.0, 3.0, -2.0),
        intensity: 50.0,
        col: Vec3::one(),
    });
    info.start_time();
    let src = unpackdb!(load_source("../Assets/Kernels/raytrace.cl"));
    info.set_time_point("Loading source file");

    let (_,_,_,program,queue) = unpackdb!(create_five(&src));
    info.set_time_point("Creating OpenCL objects");

    let (w,h) = (960u32,540u32);
    let mut kernel = unpackdb!(TraceKernel::new("raytracing", (w,h), &program, &queue, &mut scene, &mut info));
    info.set_time_point("Last time stamp");
    info.stop_time();
    info.print_info();
    unpackdb!(kernel.execute(&queue));
    let tex = unpackdb!(kernel.get_result(&queue));

    let mut window = window::Window::<state::StdState>::new("ClRays", w, h);
    window.run(window::std_input_handler, Some(tex));
    Ok(())
}
