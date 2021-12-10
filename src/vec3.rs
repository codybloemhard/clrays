#[derive(Clone, Copy, Debug)]
pub struct Orientation {
    pub yaw: f32,
    pub roll: f32
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Vec3{
    pub x: f32,
    pub y: f32,
    pub z: f32,
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

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self{
        Self { x, y, z }
    }

    #[inline]
    pub fn uni(v: f32) -> Self{
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn new_dir(ori: &Orientation) -> Self {
        let a = ori.roll;  // Up/Down
        let b = ori.yaw;   // Left/Right
        Self { x: a.cos() * b.sin(), y: a.sin(), z: a.cos() * -b.cos() }
    }

    #[inline]
    pub fn orientation(&self) -> Orientation {
        Orientation {
            yaw: f32::atan2(self.x,-self.z),
            roll: self.y.asin()
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
    pub fn less_eq(self, o: Self) -> bool{
        self.x <= o.x && self.y <= o.y && self.z <= o.z
    }
}

#[cfg(test)]
mod test{
    use crate::vec3::Vec3;
    #[test]
    fn test_new_zero(){
        assert_eq!(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);
    }
    #[test]
    fn test_new_one(){
        assert_eq!(Vec3::new(1.0, 1.0, 1.0), Vec3::ONE);
    }
    #[test]
    fn test_unharden(){
        let s = 0.1;
        let t = 1.0 - s;
        assert_eq!(Vec3::ZERO.unhardened(s), Vec3::new(s, s, s));
        assert_eq!(Vec3::ONE.unhardened(s), Vec3::new(t, t, t));
        assert_eq!(Vec3::new(0.05, 0.5, 0.3).unhardened(s), Vec3::new(s, 0.5, 0.3));
        assert_eq!(Vec3::new(0.11, 0.5, 0.89).unhardened(s), Vec3::new(0.11, 0.5, 0.89));
    }
    #[test]
    fn test_len_zero(){
        assert_eq!(Vec3::ZERO.len(), 0.0);
    }
    #[test]
    fn test_neg(){
        assert_eq!(Vec3::FORWARD.neged(), Vec3::BACKWARD);
    }
    #[test]
    fn test_len_one(){
        assert_eq!(Vec3::ONE.len() - 1.73205 < 0.001, true);
    }
    #[test]
    fn test_len_scale(){
        assert_eq!(Vec3::ONE.scaled(1.1).len() > Vec3::ONE.len(), true);
    }
    #[test]
    fn test_dot_zero(){
        assert_eq!(Vec3::ZERO.dot(Vec3::ZERO), 0.0);
    }
    #[test]
    fn test_dot_far(){
        assert_eq!(Vec3::RIGHT.dot(Vec3::UP), 0.0);
    }
    #[test]
    fn test_dot_close(){
        assert_eq!(Vec3::RIGHT.dot(Vec3::RIGHT), 1.0);
    }
    #[test]
    fn test_normalize(){
        assert_eq!((Vec3::ZERO.normalized().len()).abs() < 0.001, true);
    }
    #[test]
    fn test_normalize_fast(){
        assert_eq!((Vec3::ONE.normalized_fast().len() - 1.0).abs() < 0.001, true);
    }
    #[test]
    fn test_scale(){
        assert_eq!((Vec3::ONE.normalized_fast().scaled(5.0).len() - 5.0).abs() < 0.001, true);
    }
    #[test]
    fn test_added_0(){
        assert_eq!(Vec3::ZERO.added(Vec3::ONE), Vec3::ONE);
    }
    #[test]
    fn test_added_1(){
        assert_eq!(Vec3::ONE.added(Vec3::ZERO), Vec3::ONE);
    }
    #[test]
    fn test_added_2(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).added(Vec3::new(3.0, 2.0, 1.0)), Vec3::new(4.0, 4.0, 4.0));
    }
    #[test]
    fn test_subed_0(){
        assert_eq!(Vec3::ONE.subed(Vec3::ZERO), Vec3::ONE);
    }
    #[test]
    fn test_subed_1(){
        assert_eq!(Vec3::ONE.subed(Vec3::ONE), Vec3::ZERO);
    }
    #[test]
    fn test_subed_2(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).subed(Vec3::new(4.0, 5.0, 6.0)), Vec3::new(-3.0,-3.0,-3.0));
    }
    #[test]
    fn test_muled_0(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).muled(Vec3::new(4.0, 5.0, 6.0)), Vec3::new(4.0, 10.0, 18.0));
    }
    #[test]
    fn test_muled_1(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).muled(Vec3::ONE), Vec3::new(1.0, 2.0, 3.0));
    }
    #[test]
    fn test_dived_unsafe(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dived_fast(Vec3::new(1.0, 2.0, 3.0)), Vec3::ONE);
    }
    #[test]
    fn test_dived(){
        assert_eq!(Vec3::new(1.0, 2.0, 3.0).dived(Vec3::new(1.0, 2.0, 0.0)), Vec3::new(1.0, 1.0, std::f32::MAX));
    }
    #[test]
    fn test_crossed(){
        assert_eq!(Vec3::RIGHT.crossed(Vec3::UP), Vec3::BACKWARD);
    }
}
