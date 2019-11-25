#[macro_use] 
extern crate clrays_rs;
use clrays_rs as clr;
use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,SceneItem,Material,Plane,Sphere,Light};
use clr::vec3::{Vec3,BasicColour};
use clr::kernels::{VoidKernel,ResultKernel,TraceKernelReal};
use clr::cl_helpers::{create_five};
use clr::misc::{load_source};
use clr::info::{Info};
use clr::trace_tex::{TexType};
use clr::trace_processor::{TraceProcessor};

pub fn main() -> Result<(),String>{
    //clr::test(PlatformTest::OpenCl2);
    let mut info = Info::new();
    let mut scene = Scene::new();
    scene.sky_col = Vec3::soft_colour(BasicColour::Blue, 0.9, 0.2).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::zero();
    scene.cam_dir = Vec3::backward();
    scene.add_texture("wood", "../Assets/Textures/wood.png", TexType::Vector3c8bpc);
    scene.add_texture("sphere", "../Assets/Textures/spheremap.jpg", TexType::Vector3c8bpc);
    scene.add_texture("stone-alb", "../Assets/Textures/stone-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("stone-nor", "../Assets/Textures/stone-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("stone-rou", "../Assets/Textures/stone-rough.png", TexType::Scalar8b);
    scene.add_texture("tiles-alb", "../Assets/Textures/tiles-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("tiles-nor", "../Assets/Textures/tiles-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("tiles-rou", "../Assets/Textures/tiles-rough.png", TexType::Scalar8b);
    scene.add_texture("scifi-alb", "../Assets/Textures/scifi-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("scifi-nor", "../Assets/Textures/scifi-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("scifi-rou", "../Assets/Textures/scifi-rough.png", TexType::Scalar8b);
    scene.add_texture("scifi-met", "../Assets/Textures/scifi-metal.png", TexType::Scalar8b);
    scene.add_texture("solar-alb", "../Assets/Textures/solar-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("solar-nor", "../Assets/Textures/solar-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("solar-rou", "../Assets/Textures/solar-rough.png", TexType::Scalar8b);
    scene.add_texture("solar-met", "../Assets/Textures/solar-metal.png", TexType::Scalar8b);
    scene.add_texture("sky", "../Assets/Textures/sky1.jpg", TexType::Vector3c8bpc);
    scene.set_skybox("sky", &mut info);

    Plane{
        pos: Vec3::new(0.0, -1.0, 0.0),
        nor: Vec3::up(),
        mat: Material::basic()
            .with_texture(scene.get_texture("stone-alb", &mut info))
            .with_normal_map(scene.get_texture("stone-nor", &mut info))
            .with_roughness_map(scene.get_texture("stone-rou", &mut info))
            .with_tex_scale(4.0),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_texture(scene.get_texture("tiles-alb", &mut info))
            .with_normal_map(scene.get_texture("tiles-nor", &mut info))
            .with_roughness_map(scene.get_texture("tiles-rou", &mut info)),
    }.add(&mut scene);
    Sphere{
        pos: Vec3::new(0.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_reflectivity(0.3)
            .with_texture(scene.get_texture("solar-alb", &mut info))
            .with_normal_map(scene.get_texture("solar-nor", &mut info))
            .with_roughness_map(scene.get_texture("solar-rou", &mut info))
            .with_metalic_map(scene.get_texture("solar-met", &mut info)),
    }.add(&mut scene);
    Sphere{
        pos: Vec3::new(-2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_texture(scene.get_texture("scifi-alb", &mut info))
            .with_normal_map(scene.get_texture("scifi-nor", &mut info))
            .with_roughness_map(scene.get_texture("scifi-rou", &mut info))
            .with_metalic_map(scene.get_texture("scifi-met", &mut info))
            .with_reflectivity(0.5),
    }.add(&mut scene);
    Light{
        pos: Vec3::new(0.0, 2.0, -3.0),
        intensity: 100.0,
        col: Vec3::one(),
    }.add(&mut scene);

    //let (w,h) = (960u32,540u32);
    let (w,h) = (1600u32,900u32);

    //let mut tracer = unpackdb!(TraceProcessor::new_real((w,h), &mut scene, &mut info));
    let mut tracer = unpackdb!(TraceProcessor::new_aa((w,h), 2, &mut scene, &mut info));
    let tex = unpackdb!(tracer.render());
    
    let mut window = window::Window::<state::StdState>::new("ClRays", w, h);
    window.run(window::std_input_handler, Some(tex));
    Ok(())
}
