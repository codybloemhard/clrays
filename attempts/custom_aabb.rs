
#[derive(Clone, Copy, Debug)]
struct AABB2 {
    data: [f32;6]
}
impl AABB2{
    #[inline]
    fn surface_area(self) -> f32 {
        let v = [self.data[3] - self.data[0], self.data[4] - self.data[1], self.data[5] - self.data[2]];
        v[0] * v[1] * 2.0 + v[0] * v[2] * 2.0 + v[1] * v[2] * 2.0
    }
    #[inline]
    fn lerp_axis(self, axis: usize, percentage: f32) -> f32 {
        (1.0-percentage) * self.data[axis] + percentage * self.data[axis + 3]
    }
}

pub enum Axis { X = 0, Y = 1, Z = 2 }

fn main () {
    let bound = AABB2{data:[0.0,0.0,0.0,1.0,2.0,3.0]};
    println!("{}", bound.surface_area());
    println!("{:?}", bound.lerp_axis(0, 0.5));
    println!("{:?}", bound.lerp_axis(1, 0.5));
    println!("{:?}", bound.lerp_axis(2, 0.5));
    // println!("{:?}", bound);
}
