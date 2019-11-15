use std::num::Float;

struct vec3{
    x: f32,
    y: f32,
    z: f32,
}

impl vec3{
    fn new(x: f32, y: f32, z: f32) -> Self{
        Self {x, y, z}
    }

    fn zero() -> Self{
        Self {x: 0, y: 0, z: 0}
    }

    fn one() -> Self{
        Self {x: 1, y: 1, z: 1}
    }

    fn len(&self) -> f32{
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize_unsafe(&mut self){
        let l = self.len();
        self.x /= len;
        self.y /= len;
        self.z /= len;
    }

    fn normalize(&mut self){
        let l = self.len();
        if l == 0 { return; }
        self.x /= len;
        self.y /= len;
        self.z /= len;
    }

    fn normalized_unsafe(self) -> Self{
        self.normalize_unsafe();
        self
    }

    fn normalized(self) -> Self{
        self.normalize();
        self
    }

    fn added(self, o: &Self) -> Self{
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
        self
    }

    fn subed(self, o: &Self) -> Self{
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
        self
    }

    fn muled(self, o: &Self) -> Self{
        self.x *= o.x;
        self.y *= o.y;
        self.z *= o.z;
        self
    }

    fn dived_unsafe(self, o: &Self) -> Self{
        self.x /= o.x;
        self.y /= o.y;
        self.z /= o.z;
        self
    }

    fn dived(self, o: &Self) -> Self{
        if o.x != 0 { self.x /= o.x; }
        else { self.x = std::f32::MAX; }
        if o.y != 0 { self.y /= o.y; }
        else { self.y = std::f32::MAX; }
        if o.z != 0 { self.z /= o.z; }
        else { self.z = std::f32::MAX; }
        self
    }

    fn dot(&self, o: &Self) -> f32{
        self.x * o.x + self.y * o.y + self.z * o.z
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
        let xx = self.y * o.z - self.z * o.y;
        let yy = self.z * o.x - self.x * o.z;
        let zz = self.x * o.y - self.y * o.x;
        Self { xx, yy, zz }
    }
}
