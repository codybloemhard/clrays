#[derive(PartialEq,Clone,Copy,Debug)]
pub struct Vec3{
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub enum BasicColour{
    Red,
    Green,
    Blue,
    Black,
    White
}

impl Vec3{
    pub fn new(x: f32, y: f32, z: f32) -> Self{
        Self {x, y, z}
    }

    pub fn zero() -> Self{
        Self {x: 0.0, y: 0.0, z: 0.0}
    }

    pub fn one() -> Self{
        Self {x: 1.0, y: 1.0, z: 1.0}
    }

    pub fn right() -> Self{
        Self {x: 1.0, y: 0.0, z: 0.0}
    }

    pub fn left() -> Self{
        Self {x: -1.0, y: 0.0, z: 0.0}
    }

    pub fn up() -> Self{
        Self {x: 0.0, y: 1.0, z: 0.0}
    }

    pub fn down() -> Self{
        Self {x: 0.0, y: -1.0, z: 0.0}
    }

    pub fn forward() -> Self{
        Self {x: 0.0, y: 0.0, z: 1.0}
    }

    pub fn backward() -> Self{
        Self {x: 0.0, y: 0.0, z: -1.0}
    }

    pub fn red() -> Self{
        Self {x: 1.0, y: 0.0, z: 0.0}
    }

    pub fn green() -> Self{
        Self {x: 0.0, y: 1.0, z: 0.0}
    }

    pub fn blue() -> Self{
        Self {x: 0.0, y: 0.0, z: 1.0}
    }

    pub fn soft_colour(col: BasicColour, on_strength: f32, off_strength: f32) -> Self{
        let of = off_strength;
        let on = on_strength;
        match col{
            BasicColour::White => Self { x: on, y: on, z: on },
            BasicColour::Black => Self { x: of, y: of, z: of },
            BasicColour::Red => Self { x: on, y: of, z: of },
            BasicColour::Green => Self { x: of, y: on, z: of },
            BasicColour::Blue => Self { x: of, y: of, z: on },
        }
    }

    pub fn std_colour(col: BasicColour) -> Self{
        Vec3::soft_colour(col, 0.9, 0.1)
    }

    pub fn neg(&mut self){
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
    }

    pub fn neged(mut self) -> Self{
        self.neg();
        self
    }

    pub fn len(&self) -> f32{
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn dot(&self, o: &Self) -> f32{
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn normalize_unsafe(&mut self){
        let l = self.len();
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    pub fn normalize(&mut self){
        let l = self.len();
        if l == 0.0 { return; }
        self.x /= l;
        self.y /= l;
        self.z /= l;
    }

    pub fn normalized_unsafe(mut self) -> Self{
        self.normalize_unsafe();
        self
    }

    pub fn normalized(mut self) -> Self{
        self.normalize();
        self
    }

    pub fn scale(&mut self, s: f32){
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }

    pub fn scaled(mut self, s: f32) -> Self{
        self.scale(s);
        self
    }

    pub fn add(&mut self, o: &Self){
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }

    pub fn sub(&mut self, o: &Self){
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }

    pub fn mul(&mut self, o: &Self){
        self.x *= o.x;
        self.y *= o.y;
        self.z *= o.z;
    }

    pub fn div_unsafe(&mut self, o: &Self){
        self.x /= o.x;
        self.y /= o.y;
        self.z /= o.z;
    }

    pub fn div(&mut self, o: &Self){
        if o.x != 0.0 { self.x /= o.x; }
        else { self.x = std::f32::MAX; }
        if o.y != 0.0 { self.y /= o.y; }
        else { self.y = std::f32::MAX; }
        if o.z != 0.0 { self.z /= o.z; }
        else { self.z = std::f32::MAX; }
    }

    pub fn added(mut self, o: &Self) -> Self{
        self.add(o);
        self
    }

    pub fn subed(mut self, o: &Self) -> Self{
        self.sub(o);
        self
    }

    pub fn muled(mut self, o: &Self) -> Self{
        self.mul(o);
        self
    }

    pub fn dived_unsafe(mut self, o: &Self) -> Self{
        self.div_unsafe(o);
        self
    }

    pub fn dived(mut self, o: &Self) -> Self{
        self.div(o);
        self
    }

    pub fn cross(&mut self, o: &Self){
        let xx = self.y * o.z - self.z * o.y;
        let yy = self.z * o.x - self.x * o.z;
        let zz = self.x * o.y - self.y * o.x;
        self.x = xx;
        self.y = yy;
        self.z = zz;
    }

    pub fn crossed(self, o: &Self) -> Self{
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
    fn test_len_zero(){
        assert_eq!(Vec3::zero().len(), 0.0);
    }
    #[test]
    fn test_neg(){
        assert_eq!(Vec3::forward().neged(), Vec3::backward());
    }
    #[test]
    fn test_len_one(){
        assert_eq!(Vec3::one().len() - 1.73205 < 0.001, true);
    }
    #[test]
    fn test_len_scale(){
        assert_eq!(Vec3::one().scaled(1.1).len() > Vec3::one().len(), true);
    }
    #[test]
    fn test_dot_zero(){
        assert_eq!(Vec3::zero().dot(&Vec3::zero()), 0.0);
    }
    #[test]
    fn test_dot_far(){
        assert_eq!(Vec3::right().dot(&Vec3::up()), 0.0);
    }
    #[test]
    fn test_dot_close(){
        assert_eq!(Vec3::right().dot(&Vec3::right()), 1.0);
    }
    #[test]
    fn test_normalize(){
        assert_eq!((Vec3::zero().normalized().len()).abs() < 0.001, true);
    }
    #[test]
    fn test_normalize_unsafe(){
        assert_eq!((Vec3::one().normalized_unsafe().len() - 1.0).abs() < 0.001, true);
    }
    #[test]
    fn test_scale(){
        assert_eq!((Vec3::one().normalized_unsafe().scaled(5.0).len() - 5.0).abs() < 0.001, true);
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
    #[test]
    fn test_muled_0(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).muled(&Vec3::new(4.0, 5.0, 6.0)), Vec3::new(4.0, 10.0, 18.0));
    }
    #[test]
    fn test_muled_1(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).muled(&Vec3::one()), Vec3::new(1.0, 2.0, 3.0));
    }
    #[test]
    fn test_dived_unsafe(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dived_unsafe(&Vec3::new(1.0, 2.0, 3.0)), Vec3::one());
    }
    #[test]
    fn test_dived(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dived(&Vec3::new(1.0, 2.0, 0.0)), Vec3::new(1.0, 1.0, std::f32::MAX));
    }
    #[test]
    fn test_crossed(){
        assert_eq!(Vec3::right().crossed(&Vec3::up()), Vec3::forward());
    }
}
