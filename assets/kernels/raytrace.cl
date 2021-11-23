#define MAX_RENDER_DIST 1000000.0f
#define MAX_RENDER_DEPTH 4
#define EPSILON 0.001f
#define PI4 12.5663f
#define AMBIENT 0.001f
#define GAMMA 2.2f

#define MAT_SIZE 8
struct Material{
    float3 col;
    float reflectivity;
    float roughness;
    uint texture;
    uint normalmap;
    uint roughnessmap;
    uint metalicmap;
    float texscale;
    uchar uvtype;
};

//extract material from array, off is index of first byte of material we want
struct Material ExtractMaterial(uint off, global float *arr){
    struct Material mat;
    mat.col = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
    mat.reflectivity = arr[off + 3];
    mat.roughness = arr[off + 4] + EPSILON;
    mat.texture = (uint)arr[off + 5];
    mat.normalmap = (uint)arr[off + 6];
    mat.roughnessmap = (uint)arr[off + 7];
    mat.metalicmap  = (uint)arr[off + 8];
    mat.texscale = arr[off + 9];
    mat.uvtype = 0;
    return mat;
}

#define uvPLANE 0
#define uvSPHERE 1
#define uvBOX 2

struct RayHit{
    float3 pos;
    float3 nor;
    float t;
    struct Material *mat;
};

//hit nothing
struct RayHit NullRayHit(){
    struct RayHit hit;
    hit.t = MAX_RENDER_DIST;
    return hit;
}

struct Ray{
    float3 pos;
    float3 dir;
};

//indices for types
#define SC_LIGHT 0
#define SC_SPHERE 1
#define SC_PLANE 2
#define SC_BOX 3
#define SC_SCENE 4

struct Scene{
    global uint *params, *tex_params;
    global float *items;
    global uchar *textures;
    uint skybox;
    float3 skycol;
    float skyintens;
};

//first byte in array where this type starts
global uint ScGetStart(uint type, struct Scene *scene){
    return scene->params[type * 3 + 2];
}

//number of items of this type(not bytes!)
global uint ScGetCount(uint type, struct Scene *scene){
    return scene->params[type * 3 + 1];
}

//size of an item of this type
global uint ScGetStride(uint type, struct Scene *scene){
    return scene->params[type * 3 + 0];
}

//first byte of texture
global uint TxGetStart(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 0];
}

global uint TxGetWidth(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 1];
}

global uint TxGetHeight(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 2];
}

//get sample
global float3 TxGetSample(uint tex, struct Scene *scene, uint x, uint y, uint w){
    uint offset = TxGetStart(tex, scene) + (y * w + x) * 3;
    float3 col = (float3)(scene->textures[offset + 0],
                            scene->textures[offset + 1],
                            scene->textures[offset + 2]);
    return col / 255.0f;
}

//shared logic
#define UV_TO_XY \
    float dummy;\
    uv.x = fract(uv.x, &dummy);\
    uv.y = fract(uv.y, &dummy);\
    if(uv.x < 0.0f) uv.x += 1.0f;\
    if(uv.y < 0.0f) uv.y += 1.0f;\
    uint w = TxGetWidth(tex,scene);\
    uint x = (uint)(w * uv.x);\
    uint y = (uint)(TxGetHeight(tex,scene) * uv.y);\

//get colour from texture and uv
global float3 GetTexCol(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    return pow(TxGetSample(tex, scene, x, y, w), GAMMA);
}

//get value to range 0..1 (no gamma)
global float3 GetTexVal(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    return TxGetSample(tex, scene, x, y, w);
}

//get value 0..1 from scalar map
global float GetTexScalar(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    uint offset = TxGetStart(tex, scene) + (y * w + x);
    float scalar = (float)scene->textures[offset];
    return scalar / 255.0f;
}

//Copy a float3 out the array, off(offset) is the first byte of the float3 we want
float3 ExtractFloat3(uint off, global float *arr){
    return (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
}

float3 ExtractFloat3FromInts(global uint *arr, uint index){
    float3 res;
    res.x = as_float(arr[index + 0]);
    res.y = as_float(arr[index + 1]);
    res.z = as_float(arr[index + 2]);
    return res;
}

float3 reflect(float3 vec, float3 nor){
    return vec - 2 * (dot(vec,nor)) * nor;
}

//ray-sphere intersection
struct RayHit InterSphere(struct Ray* r, float3 spos, float srad){
    float3 l = spos - r->pos;
    float tca = dot(r->dir, l);
    float d = tca*tca - dot(l, l) + srad*srad;
    if(d < 0) return NullRayHit();
    float t = tca - sqrt(d);
    if(t < 0){
        t = tca + sqrt(d);
        if(t < 0) return NullRayHit();
    }
    struct RayHit hit;
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = (hit.pos - spos) / srad;
    return hit;
}

//ray-plane intersection
struct RayHit InterPlane(struct Ray* r, float3 ppos, float3 pnor){
    float divisor = dot(r->dir, pnor);
    if(fabs(divisor) < EPSILON) return NullRayHit();
    float3 planevec = ppos - r->pos;
    float t = dot(planevec, pnor) / divisor;
    if(t < EPSILON) return NullRayHit();
    struct RayHit hit;
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = pnor;
    return hit;
}

//ray-box intersection
//https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
//https://stackoverflow.com/questions/16875946/ray-box-intersection-normal#16876601
struct RayHit InterBox(struct Ray* r, float3 bmin, float3 bmax){
    float3 inv = 1.0f / r->dir;
    float t1 = (bmin.x - r->pos.x)*inv.x;
    float t2 = (bmax.x - r->pos.x)*inv.x;
    float t3 = (bmin.y - r->pos.y)*inv.y;
    float t4 = (bmax.y - r->pos.y)*inv.y;
    float t5 = (bmin.z - r->pos.z)*inv.z;
    float t6 = (bmax.z - r->pos.z)*inv.z;
    float tmin = fmax(fmax(fmin(t1,t2),fmin(t3,t4)),fmin(t5,t6));
    float tmax = fmin(fmin(fmax(t1,t2),fmax(t3,t4)),fmax(t5,t6));
    if(tmax < 0.0f || tmin > tmax){
        return NullRayHit();
    }
    float3 hitp = r->pos + r->dir * tmin;
    //normal
    float3 normal = (float3)(0.0f,0.0f,1.0f);
    float3 size = bmax - bmin;
    float3 localPoint = hitp - (bmin + size/2.0f);
    float min = -10000.0f;
    float distance = fabs(size.x - fabs(localPoint.x));
    if (distance < min) {
        min = distance;
        normal = (float3)(1.0f,0.0f,0.0f);
        normal *= sign(localPoint.x);
    }
    distance = fabs(size.y - fabs(localPoint.y));
    if (distance < min) {
        min = distance;
        normal = (float3)(0.0f,1.0f,0.0f);
        normal *= sign(localPoint.y);
    }
    distance = fabs(size.z - fabs(localPoint.z));
    if (distance < min) {
        min = distance;
        normal = (float3)(0.0f,0.0f,1.0f);
        normal *= sign(localPoint.z);
    }
    //return
    struct RayHit hit;
    hit.t = tmin;
    hit.pos = hitp;
    hit.nor = fast_normalize(normal);
    return hit;
}

//plane uv
float2 PlaneUV(float3 pos, float3 nor){
    float3 u = (float3)(nor.y, nor.z, -nor.x);
    float3 v = fast_normalize(cross(u, nor));
    return (float2)(dot(pos,u),dot(pos,v));
}

//sphere uv
float2 SphereUV(float3 nor){
    float u = 0.5f + (atan2(-nor.z, -nor.x) / (2*M_PI));
    float v = 0.5f - asinpi(-nor.y);
    return (float2)(u,v);
}

//sphere skybox uv(just sphere uv with inverted normal)
float2 SkySphereUV(float3 nor){
    float u = 0.5f + (atan2(nor.z, nor.x) / (2*M_PI));
    float v = 0.5f - asinpi(nor.y);
    return (float2)(u,v);
}

//macros for primitive intersections
#define START_PRIM() \
    (struct RayHit *closest, struct Ray *ray, global float *arr, const uint count, const uint start, const uint stride){\
    for(uint i = 0; i < count; i++){\
        uint off = start + i * stride;\

#define END_PRIM(offset,uvtyp) {\
            if(closest->t > hit.t){\
                *closest = hit;\
                struct Material mat = ExtractMaterial(off + offset, arr);\
                mat.uvtype = uvtyp;\
                closest->mat = &mat;\
            }\
        }\
    }\
}

//actual functions
void InterSpheres START_PRIM()
    float3 spos = ExtractFloat3(off + 0, arr);
    float srad = arr[off + 3];
    struct RayHit hit = InterSphere(ray, spos, srad);
    END_PRIM(4, uvSPHERE)

void InterPlanes START_PRIM()
    float3 ppos = ExtractFloat3(off + 0, arr);
    float3 pnor = ExtractFloat3(off + 3, arr);
    struct RayHit hit = InterPlane(ray, ppos, pnor);
    END_PRIM(6, uvPLANE)

void InterBoxes START_PRIM()
    float3 bmin = ExtractFloat3(off + 0, arr);
    float3 bmax = ExtractFloat3(off + 3, arr);
    struct RayHit hit = InterBox(ray, bmin, bmax);
    END_PRIM(6, uvBOX)

//intersect whole scene
struct RayHit InterScene(struct Ray *ray, struct Scene *scene){
    struct RayHit closest = NullRayHit();
    InterSpheres(&closest, ray, scene->items, ScGetCount(SC_SPHERE, scene), ScGetStart(SC_SPHERE, scene), ScGetStride(SC_SPHERE, scene));
    InterPlanes(&closest, ray, scene->items, ScGetCount(SC_PLANE, scene), ScGetStart(SC_PLANE, scene), ScGetStride(SC_PLANE, scene));
    InterBoxes(&closest, ray, scene->items, ScGetCount(SC_BOX, scene), ScGetStart(SC_BOX, scene), ScGetStride(SC_BOX, scene));
    return closest;
}

//get sky colour
float3 SkyCol(float3 nor, struct Scene *scene){
    if(scene->skybox == 0)
        return scene->skycol;
    float2 uv = SkySphereUV(nor);
    return GetTexCol(scene->skybox - 1, uv, scene);
}

//get diffuse light strength for hit for a light
float2 BlinnSingle(float3 lpos, float lpow, float3 viewdir, struct RayHit *hit, struct Scene *scene){
    float3 toL = lpos - hit->pos;
    float dist = fast_length(toL);
    toL /= dist + EPSILON;
    //diffuse
    float3 nor = hit->nor;
    float angle = dot(nor, toL);
    if(angle <= EPSILON)
        return (float2)(0.0f);
    angle = max(0.0f,angle);
    float power = lpow / (PI4 * dist * dist);
    if(power < 0.01f)
        return (float2)(0.0f);
    //exposed to light or not
    struct Ray lray;
    lray.pos = hit->pos + toL * EPSILON;
    lray.dir = toL;
    struct RayHit lhit = InterScene(&lray, scene);
    if(lhit.t <= dist)
        return (float2)(0.0f);
    //specular
    float3 halfdir = fast_normalize(toL + -viewdir);
    float specangle = max(dot(halfdir,nor),0.0f);
    float spec = pow(specangle,16.0f / hit->mat->roughness);
    return power * (float2)(angle,spec);
}

//get diffuse light incl colour of hit with all lights
void Blinn(struct RayHit *hit, struct Scene *scene, float3 viewdir, float3 *out_diff, float3 *out_spec){
    float3 col = (float3)(0.0f);
    float3 spec = (float3)(0.0f);
    global float* arr = scene->items;
    uint count = ScGetCount(SC_LIGHT, scene);
    uint stride = ScGetStride(SC_LIGHT, scene);
    uint start = ScGetStart(SC_LIGHT, scene);
    for(uint i = 0; i < count; i++){
        uint off = start + i * stride;
        float3 lpos = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
        float lpow = arr[off + 3];
        float3 lcol = (float3)(arr[off + 4], arr[off + 5], arr[off + 6]);
        float2 res = BlinnSingle(lpos, lpow, viewdir, hit, scene);
        col += res.x * lcol;
        spec += res.y * lcol;
    }
    //ambient
    if(scene->skyintens > EPSILON){
        struct Ray nray;
        nray.pos = hit->pos + hit->nor * EPSILON;
        nray.dir = hit->nor;
        struct RayHit nhit = InterScene(&nray, scene);
        if(nhit.t >= MAX_RENDER_DIST)
            col += SkyCol(hit->nor, scene) * scene->skyintens;
    }
    *out_diff = col * hit->mat->col;
    *out_spec = spec * (1.0f - hit->mat->roughness);
}

//Recursion only works with one function
float3 RayTrace(struct Ray *ray, struct Scene *scene, uint depth){
    if(depth == 0) return SkyCol(ray->dir, scene);

    //hit
    struct RayHit hit = InterScene(ray, scene);
    if(hit.t >= MAX_RENDER_DIST)
        return SkyCol(ray->dir, scene);

    //texture
    float2 uv;
    float3 texcol = (float3)(1.0f);
    if(hit.mat->texture > 0){
        uchar uvtype = hit.mat->uvtype;
        if(uvtype == uvPLANE)
            uv = PlaneUV(hit.pos, hit.nor);
        else if(uvtype == uvSPHERE)
            uv = SphereUV(hit.nor);
        else if(uvtype == uvBOX)
            uv = PlaneUV(hit.pos, hit.nor);
        uv *= hit.mat->texscale;
        texcol = GetTexCol(hit.mat->texture - 1, uv, scene);
    }

    //normalmap
    if(hit.mat->normalmap > 0){
        float3 rawnor = GetTexVal(hit.mat->normalmap - 1, uv, scene);
        float3 t = fast_normalize(cross(hit.nor, (float3)(0.0f,1.0f,0.0f)));
        if(fast_length(t) < EPSILON)
            t = fast_normalize(cross(hit.nor, (float3)(0.0f,0.0f,1.0f)));
        t = fast_normalize(t);
        float3 b = fast_normalize(cross(hit.nor, t));
        rawnor = rawnor * 2 - 1;
        rawnor = fast_normalize(rawnor);
        float3 newnor;
        float3 row = (float3)(t.x, b.x, hit.nor.x);
        newnor.x = dot(row, rawnor);
        row = (float3)(t.y, b.y, hit.nor.y);
        newnor.y = dot(row, rawnor);
        row = (float3)(t.z, b.z, hit.nor.z);
        newnor.z = dot(row, rawnor);
        hit.nor = fast_normalize(newnor);
    }

    //roughnessmap
    if(hit.mat->roughnessmap > 0){
        float value = GetTexScalar(hit.mat->roughnessmap - 1, uv, scene);
        hit.mat->roughness = value * hit.mat->roughness;
    }

    //metalicmap
    if(hit.mat->metalicmap > 0){
        float value = GetTexScalar(hit.mat->metalicmap - 1, uv, scene);
        hit.mat->reflectivity *= value;
    }

    //diffuse, specular
    float3 diff, spec;
    Blinn(&hit, scene, ray->dir, &diff, &spec);
    diff *= texcol;

    //reflection
    float3 newdir = fast_normalize(reflect(ray->dir, hit.nor));
    struct Ray nray;
    nray.pos = hit.pos + newdir * EPSILON;
    nray.dir = newdir;
    struct RayHit nhit = InterScene(&nray, scene);

    //Does not get corrupted to version inside recursive call if not pointer
    float refl_mul = hit.mat->reflectivity;
    float3 refl = RayTrace(&nray, scene, depth - 1);
    return (diff * (1.0f - refl_mul)) + (refl * refl_mul) + spec;
}

float3 RayTracing(const uint w, const uint h,
const uint x, const uint y, const uint AA,
__global uint *sc_params, __global float *sc_items,
__global uint *tx_params, __global uchar *tx_items){
    //Scene
    struct Scene scene;
    scene.params = sc_params;
    scene.items = sc_items;
    scene.tex_params = tx_params;
    scene.textures = tx_items;
    scene.skybox = sc_params[3*SC_SCENE + 0];
    scene.skycol = ExtractFloat3FromInts(sc_params, 3*SC_SCENE + 1);
    scene.skyintens = as_float(sc_params[3*SC_SCENE + 4]);
    struct Ray ray;
    ray.pos = ExtractFloat3FromInts(sc_params, 3*SC_SCENE + 5);
    float3 cd = fast_normalize(ExtractFloat3FromInts(sc_params, 3*SC_SCENE + 8));
    float3 hor = fast_normalize(cross(cd,(float3)(0.0f,1.0f,0.0f)));
    float3 ver = fast_normalize(cross(hor,cd));
    float2 uv = (float2)((float)x / (w * AA), (float)y / (h * AA));
    uv -= 0.5f;
    uv *= (float2)((float)w/h, -1.0f);
    float3 to = ray.pos + cd;
    to += uv.x * hor;
    to += uv.y * ver;
    ray.dir = fast_normalize(to - ray.pos);

    float3 col = RayTrace(&ray, &scene, MAX_RENDER_DEPTH);
    col = pow(col, (float3)(1.0f/GAMMA));
    if(AA == 1)
        col = clamp(col,0.0f,1.0f);
    col /= (float)(AA*AA);
    return col;
}

//https://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
void AtomicFloatAdd(volatile global float *source, const float operand) {
    union { uint intVal; float floatVal; } newVal;
    union { uint intVal; float floatVal; } prevVal;
    do{
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    }
    while (atomic_cmpxchg((volatile global uint *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void raytracingAA(
    __global float *floatmap,
    const uint w,
    const uint h,
    const uint AA,
    __global  int *sc_params,
    __global float *sc_items,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = ((x/AA) + ((y/AA) * w)) * 3;
    float3 col = RayTracing(w, h, x, y, AA,
        sc_params, sc_items, tx_params, tx_items);
    AtomicFloatAdd(&floatmap[pixid + 0],col.x);
    AtomicFloatAdd(&floatmap[pixid + 1],col.y);
    AtomicFloatAdd(&floatmap[pixid + 2],col.z);
}

__kernel void raytracing(
    __global uint *intmap,
    const uint w,
    const uint h,
    __global uint *sc_params,
    __global float *sc_items,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = x + y * w;
    float3 col = RayTracing(w, h, x, y, 1,
        sc_params, sc_items, tx_params, tx_items);
    col *= 255;
    uint res = ((uint)col.x << 16) + ((uint)col.y << 8) + (uint)col.z;
    intmap[pixid] = res;
}

//takes same input as raytracing, outputs a gradient
__kernel void raytracing_format_gradient_test(
    __global uint *intmap,
    const uint w,
    const uint h,
    __global uint *sc_params,
    __global float *sc_items,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = x + y * w;
    float3 col = (float3)((float)x / w,(float)y / h, 0.0);
    col *= 255.0f;
    uint res = ((uint)col.x << 16) + ((uint)col.y << 8) + (uint)col.z;
    intmap[pixid] = res;
}

//takes same input as raytracing, outputs the first texture
__kernel void raytracing_format_texture_test(
    __global uint *intmap,
    const uint w,
    const uint h,
    __global uint *sc_params,
    __global float *sc_items,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = x + y * w;
    uint ww = tx_params[1];
    uint hh = tx_params[2];
    uint xx = (float)x / w * ww;
    uint yy = (float)y / h * hh;
    uchar rr = tx_items[(yy * ww + xx) * 3 + 0];
    uchar gg = tx_items[(yy * ww + xx) * 3 + 1];
    uchar bb = tx_items[(yy * ww + xx) * 3 + 2];
    uint res = ((uint)rr << 16) + ((uint)gg << 8) + (uint)bb;
    intmap[pixid] = res;
}

__kernel void clear(
    __global float *floatmap,
    const uint w,
    const uint h
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = (x + y * w) * 3;
    floatmap[pixid + 0] = 0.0f;
    floatmap[pixid + 1] = 0.0f;
    floatmap[pixid + 2] = 0.0f;
}

__kernel void image_from_floatmap(
    __global float *floatmap,
    __global uint *imagemap,
    const uint w,
    const uint h
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pix_int = (x + y * w);
    uint pix_float = pix_int * 3;
    float r = clamp(floatmap[pix_float + 0], 0.0f, 1.0f) * 255.0f;
    float g = clamp(floatmap[pix_float + 1], 0.0f, 1.0f) * 255.0f;
    float b = clamp(floatmap[pix_float + 2], 0.0f, 1.0f) * 255.0f;
    imagemap[pix_int] = ((uint)((uchar)r) << 16) + ((uint)((uchar)g) << 8) + (uint)((uchar)b);
}
