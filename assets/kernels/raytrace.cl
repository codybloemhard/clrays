#define MAX_RENDER_DIST 1000000.0f
#define MAX_RENDER_DEPTH 4
#define EPSILON 0.0001f
#define PI4 12.5663f
#define AMBIENT 0.05f
#define GAMMA 2.2f

struct Material{
    float3 col;
    float reflectivity;
    float roughness;
    uint texture;
    uint normalmap;
    uint roughnessmap;
    uint metalicmap;
    float texscale;
};

#define uvPLANE 0
#define uvSPHERE 1

struct RayHit{
    float3 pos;
    float3 nor;
    float t;
    uint mat_index;
    uchar uvtype;
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
#define SC_MAT 0
#define SC_LIGHT 1
#define SC_PLANE 2
#define SC_SPHERE 3
#define SC_TRI 4
#define SC_SCENE 5

struct Scene{
    global uint *params, *tex_params;
    global float *items;
    global uchar *textures;
    global uint *bvh;
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
    return mat;
}

struct Material GetMaterialFromIndex(uint index, struct Scene *scene){
    uint start = ScGetStart(SC_MAT, scene);
    uint stride = ScGetStride(SC_MAT, scene);
    return ExtractMaterial(start + index * stride, scene->items);
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
    return vec - 2 * (dot(vec, nor)) * nor;
}

//ray-sphere intersection
bool InterSphere(struct Ray* r, struct RayHit* hit, float3 spos, float srad){
    float3 l = spos - r->pos;
    float tca = dot(r->dir, l);
    float d = tca*tca - dot(l, l) + srad*srad;
    if(d < 0) return false;
    float t = tca - sqrt(d);
    if(t < 0){
        t = tca + sqrt(d);
        if (t < 0) return false;
    }
    if(t >= hit->t) return false;
    hit->t = t;
    hit->pos = r->pos + r->dir * t;
    hit->nor = (hit->pos - spos) / srad;
    return true;
}

//ray-plane intersection
bool InterPlane(struct Ray* r, struct RayHit* hit, float3 ppos, float3 pnor){
    float divisor = dot(r->dir, pnor);
    if(fabs(divisor) < EPSILON) return false;
    float3 planevec = ppos - r->pos;
    float t = dot(planevec, pnor) / divisor;
    if(t < EPSILON) return false;
    if(t >= hit->t) return false;
    hit->t = t;
    hit->pos = r->pos + r->dir * t;
    hit->nor = pnor;
    return true;
}

bool InterTri(struct Ray* r, struct RayHit* hit, float3 ta, float3 tb, float3 tc){
    float3 edge1 = tb - ta;
    float3 edge2 = tc - ta;
    float3 h = cross(r->dir, edge2);
    float a = dot(edge1, h);
    if(a > -EPSILON * 0.01 && a < EPSILON * 0.01) return false;
    float f = 1.0 / a;
    float3 s = r->pos - ta;
    float u = f * dot(s, h);
    if(u < 0.0 || u > 1.0) return false;
    float3 q = cross(s, edge1);
    float v = f * dot(r->dir, q);
    if(v < 0.0 || u + v > 1.0) return false;
    float t = f * dot(edge2, q);
    if(t <= EPSILON) return false;
    if(t >= hit->t) return false;
    hit->t = t;
    hit->pos = r->pos + r->dir * t;
    hit->nor = fast_normalize(cross(edge1, edge2));
    return true;
}

// ray-box intersection
// https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
// https://stackoverflow.com/questions/16875946/ray-box-intersection-normal#16876601
float InterAABB(struct Ray* r, float3 bmin, float3 bmax){
    float3 inv = 1.0f / r->dir;
    float t1 = (bmin.x - r->pos.x) * inv.x;
    float t2 = (bmax.x - r->pos.x) * inv.x;
    float t3 = (bmin.y - r->pos.y) * inv.y;
    float t4 = (bmax.y - r->pos.y) * inv.y;
    float t5 = (bmin.z - r->pos.z) * inv.z;
    float t6 = (bmax.z - r->pos.z) * inv.z;
    float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
    float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));
    if(tmax < 0.0f || tmin > tmax){
        return MAX_RENDER_DIST;
    }
    return tmin;
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
    uint coff = UINT_MAX;\
    for(uint i = 0; i < count; i++){\
        uint off = start + i * stride;\

#define END_PRIM(mat_prim_offset, uvtyp) {\
            if(hit) coff = off;\
        }\
        if(coff != UINT_MAX){\
            closest->mat_index = arr[coff + mat_prim_offset];\
            closest->uvtype = uvtyp;\
        }\
    }\
}\

//actual functions
void InterSpheres START_PRIM()
    float3 spos = ExtractFloat3(off + 0, arr);
    float srad = arr[off + 3];
    bool hit = InterSphere(ray, closest, spos, srad);
END_PRIM(4, uvSPHERE)

void InterPlanes START_PRIM()
    float3 ppos = ExtractFloat3(off + 0, arr);
    float3 pnor = ExtractFloat3(off + 3, arr);
    bool hit = InterPlane(ray, closest, ppos, pnor);
END_PRIM(6, uvPLANE)

void InterTris START_PRIM()
    float3 a = ExtractFloat3(off + 0, arr);
    float3 b = ExtractFloat3(off + 3, arr);
    float3 c = ExtractFloat3(off + 6, arr);
    bool hit = InterTri(ray, closest, a, b, c);
END_PRIM(9, uvPLANE)

//intersect whole scene
struct RayHit InterScene(struct Ray *ray, struct Scene *scene){
    struct RayHit closest = NullRayHit();
    InterPlanes(&closest, ray, scene->items, ScGetCount(SC_PLANE, scene), ScGetStart(SC_PLANE, scene), ScGetStride(SC_PLANE, scene));
    InterSpheres(&closest, ray, scene->items, ScGetCount(SC_SPHERE, scene), ScGetStart(SC_SPHERE, scene), ScGetStride(SC_SPHERE, scene));
    InterTris(&closest, ray, scene->items, ScGetCount(SC_TRI, scene), ScGetStart(SC_TRI, scene), ScGetStride(SC_TRI, scene));
    return closest;
}

void InterBvh(struct Ray *ray, struct Scene *scene, struct RayHit *closest, uint current, uint index_start, uint vertex_start){
    // uint stack[64];
    // uint* ptr = stack;
    // *ptr++ = NULL;
    // uint current = 0;
    //
    // do{
    //
    // }
    // while();

    uint v = vertex_start + current * 8;
    float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
    float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
    uint left_first = scene->bvh[vertex_start + 6];
    uint count = scene->bvh[vertex_start + 7];

    if(count > 0){ // leaf
        uint sph_start = ScGetStart(SC_SPHERE, scene);
        uint sph_stride = ScGetStride(SC_SPHERE, scene);
        uint sph_count = ScGetCount(SC_SPHERE, scene);
        uint tri_start = ScGetStart(SC_TRI, scene);
        uint tri_stride = ScGetStride(SC_TRI, scene);

        uchar uvtype = 0;
        uint coff = UINT_MAX;

        for(uint i = left_first; i < left_first + count; i++){
            uint k = scene->bvh[2 + i]; // 2 = index_start for now
            if(k < sph_count){ // sphere
                uint off = sph_start + k * sph_stride;
                float3 spos = ExtractFloat3(off + 0, scene->items);
                float srad = scene->items[off + 3];
                bool hit = InterSphere(ray, closest, spos, srad);
                // bool hit = false;
                if(hit){
                    coff = off;
                    uvtype = uvSPHERE;
                }
            } else { // triangle
                uint off = tri_start + (k - sph_count) * tri_stride;
                float3 a = ExtractFloat3(off + 0, scene->items);
                float3 b = ExtractFloat3(off + 3, scene->items);
                float3 c = ExtractFloat3(off + 6, scene->items);
                bool hit = InterTri(ray, closest, a, b, c);
                // bool hit = false;
                if(hit){
                    coff = off;
                    uvtype = uvPLANE;
                }
            }
        }

        // InterSpheres(closest, ray, scene->items, ScGetCount(SC_SPHERE, scene), ScGetStart(SC_SPHERE, scene), ScGetStride(SC_SPHERE, scene));
        // InterTris(closest, ray, scene->items, ScGetCount(SC_TRI, scene), ScGetStart(SC_TRI, scene), ScGetStride(SC_TRI, scene));

        uint mat_prim_offset = uvtype == uvSPHERE ? 4 : 9;
        if(coff != UINT_MAX){
            closest->mat_index = scene->items[coff + mat_prim_offset];
            closest->uvtype = uvtype;
        }
    } else {
        uint vertices[2] = {
            left_first,
            left_first + 1,
        };

        v = vertex_start + left_first * 8;
        float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
        float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
        float t0 = InterAABB(ray, bmin, bmax);

        v = vertex_start + (left_first + 1) * 8;
        bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
        bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
        float t1 = InterAABB(ray, bmin, bmax);
        float ts[2] = { t0, t1 };

        uint order[2] = { 0, 0 };
        if(ts[0] <= ts[1]){
            order[1] = 1;
        } else {
            order[0] = 1;
        }

        if(ts[order[0]] < closest->t){
            InterBvh(ray, scene, closest, vertices[order[0]], index_start, vertex_start);
            if(ts[order[1]] < closest->t){
                InterBvh(ray, scene, closest, vertices[order[1]], index_start, vertex_start);
            }
        }
    }
}

struct RayHit InterSceneBvh(struct Ray *ray, struct Scene *scene){
    struct RayHit closest = NullRayHit();
    uint index_start = scene->bvh[0];
    uint vertex_start = scene->bvh[1];
    InterBvh(ray, scene, &closest, 0, index_start, vertex_start);
    return closest;
}

// #define INTER_SCENE InterScene
#define INTER_SCENE InterSceneBvh

float BvhDebug(struct Ray *ray, struct Scene *scene, uint current, uint vertex_start, uint d){
    uint v = vertex_start + current * 8;
    float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
    float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
    uint left_first = scene->bvh[vertex_start + 6];
    uint count = scene->bvh[vertex_start + 7];

    if(count > 0){ // leaf
        return 1.0;
    } else {
        uint vertices[2] = {
            left_first,
            left_first + 1,
        };

        v = vertex_start + left_first * 8;
        float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
        float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
        float t0 = InterAABB(ray, bmin, bmax);

        v = vertex_start + (left_first + 1) * 8;
        bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
        bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
        float t1 = InterAABB(ray, bmin, bmax);
        float ts[2] = { t0, t1 };

        uint order[2] = { 0, 0 };
        if(ts[0] <= ts[1]){
            order[1] = 1;
        } else {
            order[0] = 1;
        }
        if(d > 4) return 1.0;

        float x0 = 0.0;
        float x1 = 0.0;
        if(t0 != MAX_RENDER_DIST) x0 = BvhDebug(ray, scene, vertices[order[0]], vertex_start, d + 1);
        if(t1 != MAX_RENDER_DIST) x1 = BvhDebug(ray, scene, vertices[order[1]], vertex_start, d + 1);

        return 2.0 + x0 + x1;

        // if(t0 == MAX_RENDER_DIST && t1 == MAX_RENDER_DIST) return 0.0;
        // else return 20.0;
    }
}

//get sky colour
float3 SkyCol(float3 nor, struct Scene *scene){
    if(scene->skybox == 0)
        return scene->skycol;
    float2 uv = SkySphereUV(nor);
    return GetTexCol(scene->skybox - 1, uv, scene);
}

//get diffuse light strength for hit for a light
float2 BlinnSingle(float3 lpos, float lpow, float3 viewdir, float roughness, struct RayHit *hit, struct Scene *scene){
    float3 toL = lpos - hit->pos;
    float dist = fast_length(toL);
    toL /= dist + EPSILON;
    //diffuse
    float3 nor = hit->nor;
    float angle = dot(nor, toL);
    if(angle <= EPSILON)
        return (float2)(0.0f);
    angle = max(0.0f, angle);
    float power = lpow / (PI4 * dist * dist);
    if(power < 0.01f)
        return (float2)(0.0f);
    //exposed to light or not
    struct Ray lray;
    lray.pos = hit->pos + hit->nor * EPSILON;
    lray.dir = toL;
    struct RayHit lhit = INTER_SCENE(&lray, scene);
    if(lhit.t <= dist)
        return (float2)(0.0f);
    //specular
    float3 halfdir = fast_normalize(toL + -viewdir);
    float specangle = max(dot(halfdir,nor),0.0f);
    float spec = pow(specangle,16.0f / roughness);
    return power * (float2)(angle,spec);
}

//get diffuse light incl colour of hit with all lights
void Blinn(struct RayHit *hit, struct Scene *scene, float3 viewdir, float3 colour, float roughness, float3 *out_diff, float3 *out_spec){
    float3 col = (float3)(AMBIENT);
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
        float2 res = BlinnSingle(lpos, lpow, viewdir, roughness, hit, scene);
        col += res.x * lcol;
        spec += res.y * lcol;
    }
    *out_diff = col * colour;
    *out_spec = spec * (1.0f - roughness);
}

//Recursion only works with one function
float3 RayTrace(struct Ray *ray, struct Scene *scene, uint depth){
    float hits = BvhDebug(ray, scene, 0, scene->bvh[1], 0);
    return (float3)(1.0) * (hits / 30);
    if(depth == 0) return SkyCol(ray->dir, scene);

    //hit
    struct RayHit hit = INTER_SCENE(ray, scene);
    if(hit.t >= MAX_RENDER_DIST)
        return SkyCol(ray->dir, scene);

    //texture
    float2 uv;
    float3 texcol = (float3)(1.0f);
    struct Material mat = GetMaterialFromIndex(hit.mat_index, scene);
    // if (hit.mat_index == 1){
    //     return (float3)(0.0, 0.0, 0.0);
    // }
    if(mat.texture > 0){
        uchar uvtype = hit.uvtype;
        if(uvtype == uvPLANE)
            uv = PlaneUV(hit.pos, hit.nor);
        else if(uvtype == uvSPHERE)
            uv = SphereUV(hit.nor);
        uv *= mat.texscale;
        texcol = GetTexCol(mat.texture - 1, uv, scene);
    }

    //normalmap
    if(mat.normalmap > 0){
        float3 rawnor = GetTexVal(mat.normalmap - 1, uv, scene);
        float3 t = cross(hit.nor, (float3)(0.0f,1.0f,0.0f));
        if(fast_length(t) < EPSILON)
            t = cross(hit.nor, (float3)(0.0f,0.0f,1.0f));
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
    if(mat.roughnessmap > 0){
        float value = GetTexScalar(mat.roughnessmap - 1, uv, scene);
        mat.roughness = value * mat.roughness;
    }

    //metalicmap
    if(mat.metalicmap > 0){
        float value = GetTexScalar(mat.metalicmap - 1, uv, scene);
        mat.reflectivity *= value;
    }

    //diffuse, specular
    float3 diff, spec;
    Blinn(&hit, scene, ray->dir, mat.col, mat.roughness, &diff, &spec);
    diff *= texcol;

    //reflection
    float3 newdir = fast_normalize(reflect(ray->dir, hit.nor));
    struct Ray nray;
    // using hit.nor instead of newdir slows down by 5ms???
    nray.pos = hit.pos + newdir * EPSILON;
    nray.dir = newdir;
    struct RayHit nhit = INTER_SCENE(&nray, scene);

    // Does not get corrupted to version inside recursive call if not pointer
    float refl_mul = mat.reflectivity;
    float3 refl = RayTrace(&nray, scene, depth - 1);
    return (diff * (1.0f - refl_mul)) + (refl * refl_mul) + spec;
}

float3 RayTracing(const uint w, const uint h,
const uint x, const uint y, const uint AA,
__global uint *sc_params, __global float *sc_items,
__global uint *tx_params, __global uchar *tx_items,
__global uint *bvh){
    //Scene
    struct Scene scene;
    scene.params = sc_params;
    scene.items = sc_items;
    scene.tex_params = tx_params;
    scene.textures = tx_items;
    scene.bvh = bvh;
    scene.skybox = sc_params[3 * SC_SCENE + 0];
    scene.skycol = ExtractFloat3FromInts(sc_params, 3 * SC_SCENE + 1);
    scene.skyintens = as_float(sc_params[3 * SC_SCENE + 4]);
    struct Ray ray;
    ray.pos = ExtractFloat3FromInts(sc_params, 3 * SC_SCENE + 5);
    float3 cd = fast_normalize(ExtractFloat3FromInts(sc_params, 3 * SC_SCENE + 8));
    float3 hor = fast_normalize(cross(cd, (float3)(0.0f, 1.0f, 0.0f)));
    float3 ver = fast_normalize(cross(hor,cd));
    float2 uv = (float2)((float)x / (w * AA), (float)y / (h * AA));
    uv -= 0.5f;
    uv *= (float2)((float)w / h, -1.0f);
    float3 to = ray.pos + cd;
    to += uv.x * hor;
    to += uv.y * ver;
    ray.dir = fast_normalize(to - ray.pos);

    float3 col = RayTrace(&ray, &scene, MAX_RENDER_DEPTH);
    col = pow(col, (float3)(1.0f / GAMMA));
    if(AA == 1)
        col = clamp(col, 0.0f, 1.0f);
    col /= (float)(AA * AA);
    return col;
}

//https://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
// void AtomicFloatAdd(volatile global float *source, const float operand) {
//     union { uint intVal; float floatVal; } newVal;
//     union { uint intVal; float floatVal; } prevVal;
//     do{
//         prevVal.floatVal = *source;
//         newVal.floatVal = prevVal.floatVal + operand;
//     }
//     while (atomic_cmpxchg((volatile global uint *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
// }

// __kernel void raytracingAA(
//     __global float *floatmap,
//     const uint w,
//     const uint h,
//     const uint AA,
//     __global  int *sc_params,
//     __global float *sc_items,
//     __global uint *tx_params,
//     __global uchar *tx_items
// ){
//     uint x = get_global_id(0);
//     uint y = get_global_id(1);
//     uint pixid = ((x/AA) + ((y/AA) * w)) * 3;
//     float3 col = RayTracing(w, h, x, y, AA,
//         sc_params, sc_items, tx_params, tx_items);
//     AtomicFloatAdd(&floatmap[pixid + 0],col.x);
//     AtomicFloatAdd(&floatmap[pixid + 1],col.y);
//     AtomicFloatAdd(&floatmap[pixid + 2],col.z);
// }

__kernel void raytracing(
    __global uint *intmap,
    const uint w,
    const uint h,
    __global uint *sc_params,
    __global float *sc_items,
    __global uint *bvh,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = x + y * w;
    float3 col = RayTracing(w, h, x, y, 1,
        sc_params, sc_items, tx_params, tx_items, bvh);
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
