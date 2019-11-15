#[derive(PartialEq,Clone,Copy,Debug)]
struct Vec3{
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3{
    fn new(x: f32, y: f32, z: f32) -> Self{
        Self {x, y, z}
    }

    fn zero() -> Self{
        Self {x: 0.0, y: 0.0, z: 0.0}
    }

    fn one() -> Self{
        Self {x: 1.0, y: 1.0, z: 1.0}
    }

    fn len(&self) -> f32{
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn dot(&self, o: &Self) -> f32{
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn normalize_unsafe(&mut self){
        let l = self.len();
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    fn normalize(&mut self){
        let l = self.len();
        if l == 0.0 { return; }
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    fn normalized_unsafe(mut self) -> Self{
        self.normalize_unsafe();
        self
    }

    fn normalized(mut self) -> Self{
        self.normalize();
        self
    }

    fn scale(&mut self, s: f32){
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }

    fn scaled(mut self, s: f32) -> Self{
        self.scale(s);
        self
    }

    fn added(mut self, o: &Self) -> Self{
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
        self
    }

    fn subed(mut self, o: &Self) -> Self{
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
        self
    }

    fn muled(mut self, o: &Self) -> Self{
        self.x *= o.x;
        self.y *= o.y;
        self.z *= o.z;
        self
    }

    fn dived_unsafe(mut self, o: &Self) -> Self{
        self.x /= o.x;
        self.y /= o.y;
        self.z /= o.z;
        self
    }

    fn dived(mut self, o: &Self) -> Self{
        if o.x != 0.0 { self.x /= o.x; }
        else { self.x = std::f32::MAX; }
        if o.y != 0.0 { self.y /= o.y; }
        else { self.y = std::f32::MAX; }
        if o.z != 0.0 { self.z /= o.z; }
        else { self.z = std::f32::MAX; }
        self
    }

    fn cross(&mut self, o: &Self){
        let xx = self.y * o.z - self.z * o.y;
        let yy = self.z * o.x - self.x * o.z;
        let zz = self.x * o.y - self.y * o.x;
        self.x = xx;
        self.y = yy;
        self.z = zz;
    }
    
    fn crossed(self, o: &Self) -> Self{
        let x = self.y * o.z - self.z * o.y;
        let y = self.z * o.x - self.x * o.z;
        let z = self.x * o.y - self.y * o.x;
        Self { x, y, z }
    }
}

#[cfg(test)]
mod test{
    use crate::vec3::Vec3;
    #[test]
    fn test_new_zero(){
        assert_eq!(Vec3::new(0.0, 0.0, 0.0), Vec3::zero());
    }
    #[test]
    fn test_new_one(){
        assert_eq!(Vec3::new(1.0, 1.0, 1.0), Vec3::one());
    }
    #[test]
    fn test_added_0(){
        assert_eq!(Vec3::zero().added(&Vec3::one()), Vec3::one());
    }
    #[test]
    fn test_added_1(){
        assert_eq!(Vec3::one().added(&Vec3::zero()), Vec3::one());
    }
    #[test]
    fn test_added_2(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).added(&Vec3::new(3.0, 2.0, 1.0)), Vec3::new(4.0, 4.0, 4.0));
    }
    #[test]
    fn test_subed_0(){
        assert_eq!(Vec3::one().subed(&Vec3::zero()), Vec3::one());
    }
    #[test]
    fn test_subed_1(){
        assert_eq!(Vec3::one().subed(&Vec3::one()), Vec3::zero());
    }
    #[test]
    fn test_subed_2(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).subed(&Vec3::new(4.0, 5.0, 6.0)), Vec3::new(-3.0,-3.0,-3.0));
    }
}
