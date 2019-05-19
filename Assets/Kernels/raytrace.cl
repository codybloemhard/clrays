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
    float shininess;
    int texture;
    int normalmap;
    float texscale;
    uchar uvtype;
};
//extract material from array, off is index of first byte of material we want
struct Material ExtractMaterial(int off, global float *arr){
    struct Material mat;
    mat.col = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
    mat.reflectivity = arr[off + 3];
    mat.shininess = arr[off + 4];
    mat.texture = (int)arr[off + 5];
    mat.normalmap = (int)arr[off + 6];
    mat.texscale = arr[off + 7];
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
    global int *params, *tex_params;
    global float *items;
    global uchar *textures;
    int skybox;
    float3 skycol;
    float skyintens;
};
//first byte in array where this type starts
global int ScGetStart(int type, struct Scene *scene){
    return scene->params[type * 3 + 2];
}
//number of items of this type(not bytes!)
global int ScGetCount(int type, struct Scene *scene){
    return scene->params[type * 3 + 1];
}
//size of an item of this type
global int ScGetStride(int type, struct Scene *scene){
    return scene->params[type * 3 + 0];
}
//first byte of texture
global int TxGetStart(int tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 0];
}
global int TxGetWidth(int tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 1];
}
global int TxGetHeight(int tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 2];
}
//get sample
global float3 TxGetSample(int tex, struct Scene *scene, int x, int y, int w){
    int offset = TxGetStart(tex, scene) + (y * w + x) * 3;
    float3 col = (float3)(scene->textures[offset + 0],
                            scene->textures[offset + 1],
                            scene->textures[offset + 2]);
    return pow(col.zyx / 256.0f, GAMMA);
    //return col.zyx / 256.0f;
}
//get colour from texture and uv
global float3 GetTexCol(int tex, float2 uv, struct Scene *scene){
    float dummy;
    uv.x = fract(uv.x, &dummy);
    uv.y = fract(uv.y, &dummy);
    if(uv.x < 0.0f) uv.x += 1.0f;
    if(uv.y < 0.0f) uv.y += 1.0f;
    int w = TxGetWidth(tex,scene);
    int x = (int)(w * uv.x);
    int y = (int)(TxGetHeight(tex,scene) * uv.y);
    return TxGetSample(tex, scene, x, y, w);
}
//Copy a float3 out the array, off(offset) is the first byte of the float3 we want
float3 ExtractFloat3(int off, global float *arr){
    return (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
}

float3 ExtractFloat3FromInts(global int *arr, int index){
    float3 res;
    res.x = as_float(arr[index + 0]);
    res.y = as_float(arr[index + 1]);
    res.z = as_float(arr[index + 2]);
    return res;
}

float dist2(float3 a, float3 b){
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
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
//plane uv
float2 PlaneUV(float3 pos, float3 nor){
    float3 u = normalize((float3)(nor.y, nor.z, -nor.x));
    float3 v = normalize(cross(u, nor));
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
//ray-box intersection
//https://stackoverflow.com/questions/16875946/ray-box-intersection-normal#16876601
//https://tavianator.com/fast-branchless-raybounding-box-intersections/
//broken
struct RayHit InterBox(struct Ray* r, float3 bmin, float3 bmax){
    float3 inv = 1.0f / r->dir;
    float tx1 = (bmin.x - r->pos.x)*inv.x;
    float tx2 = (bmax.x - r->pos.x)*inv.x;
 
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
 
    float ty1 = (bmin.y - r->pos.y)*inv.y;
    float ty2 = (bmax.y - r->pos.y)*inv.y;
 
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (bmin.z - r->pos.z)*inv.z;
    float tz2 = (bmax.z - r->pos.z)*inv.z;
 
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
 
    if(tmax < tmin)
        return NullRayHit();

    float t = max(tmin,0.0f);
    float3 hitp = r->pos + r->dir * t;
    //normal
    float3 normal;
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
    hit.t = t;
    hit.pos = hitp;
    hit.nor = normalize(normal);
    return hit;
}
//macros for primitive intersections
#define START_PRIM() \
    (struct RayHit *closest, struct Ray *ray, global float *arr, const int count, const int start, const int stride){\
    for(int i = 0; i < count; i++){\
        int off = start + i * stride;\

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
    float dist = length(toL);
    toL /= dist + EPSILON;
    //diffuse
    float3 nor = hit->nor;
    float angle = dot(nor, toL);
    if(angle <= EPSILON)
        return (float2)(0.0f);
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
    float3 halfdir = normalize(toL + -viewdir);
    float specangle = max(dot(halfdir,nor),0.0f);
    float spec = pow(specangle,hit->mat->shininess);
    return power * (float2)(max(0.0f, angle),spec);
}
//get diffuse light incl colour of hit with all lights
void Blinn(struct RayHit *hit, struct Scene *scene, float3 viewdir, float3 *out_diff, float3 *out_spec){
    float3 col = (float3)(0.0f);
    float3 spec = (float3)(0.0f);
    global float* arr = scene->items;
    int count = ScGetCount(SC_LIGHT, scene);
    int stride = ScGetStride(SC_LIGHT, scene);
    int start = ScGetStart(SC_LIGHT, scene);
    for(int i = 0; i < count; i++){
        int off = start + i * stride;
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
    *out_spec = spec;
}
//Recursion only works with one function
float3 RayTrace(struct Ray *ray, struct Scene *scene, int depth){
    if(depth == 0) return SkyCol(ray->dir, scene);
    //hit
    struct RayHit hit = InterScene(ray, scene);
    if(hit.t >= MAX_RENDER_DIST)
        return SkyCol(ray->dir, scene);
    //texture
    float2 uv;
    float3 texcol;
    if(hit.mat->texture > 0){
        uchar uvtype = hit.mat->uvtype;
        if(uvtype == uvPLANE)
            uv = PlaneUV(hit.pos, hit.nor);
        else if(uvtype == uvSPHERE)
            uv = SphereUV(hit.nor);
        uv *= hit.mat->texscale;
        texcol = GetTexCol(hit.mat->texture - 1, uv, scene);
    }
    //normalmap
    if(hit.mat->normalmap > 0){
        float3 rawnor = GetTexCol(hit.mat->normalmap - 1, uv, scene);
        float3 t = cross(hit.nor, (float3)(0.0f,1.0f,0.0f));
        if(length(t) < EPSILON)
            t = cross(hit.nor, (float3)(0.0f,0.0f,1.0f));
        t = normalize(t);
        float3 b = normalize(cross(hit.nor, t));
        rawnor = normalize(rawnor);
        rawnor = rawnor * 2 - 1;
        float3 newnor;
        float3 row = (float3)(t.x, b.x, hit.nor.x);
        newnor.x = dot(row, rawnor);
        row = (float3)(t.y, b.y, hit.nor.y);
        newnor.y = dot(row, rawnor);
        row = (float3)(t.z, b.z, hit.nor.z);
        newnor.z = dot(row, rawnor);
        hit.nor = normalize(newnor);
    }
    //diffuse
    float3 diff, spec;
    Blinn(&hit, scene, ray->dir, &diff, &spec);
    diff *= texcol;
    //reflection
    float3 newdir = normalize(reflect(ray->dir, hit.nor));
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
const int x, const int y, const uint AA,
__global int *sc_params, __global float *sc_items,
__global int *tx_params, __global uchar *tx_items){
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
    float3 cd = ExtractFloat3FromInts(sc_params, 3*SC_SCENE + 8);
    float3 hor = cross(cd,(float3)(0.0f,1.0f,0.0f));
    float3 ver = cross(hor,cd);
    float2 uv = (float2)((float)x / (w * AA), (float)y / (h * AA));
    uv -= 0.5f;
    uv *= (float2)((float)w/h, -1.0f);
    float3 to = ray.pos + cd;
    to += uv.x * hor;
    to += uv.y * ver;
    ray.dir = normalize(to - ray.pos);

    float3 col = RayTrace(&ray, &scene, MAX_RENDER_DEPTH);
    col = pow(col, (float3)(1.0f/GAMMA));
    if(AA == 1) col = clamp(col,0.0f,1.0f);
    col /= AA;
    return col;
}

__kernel void raytracingAA(
    __global float *floatmap,
    const uint w,
    const uint h,
    const uint AA,
    __global int *sc_params,
    __global float *sc_items,
    __global int *tx_params,
    __global uchar *tx_items
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = (x/AA + y/AA * w) * 3;
    float3 col = RayTracing(w, h, x, y, AA,
        sc_params, sc_items, tx_params, tx_items);
    floatmap[pixid + 0] += col.x;
    floatmap[pixid + 1] += col.y;
    floatmap[pixid + 2] += col.z;
}

__kernel void raytracing(
    __global int *intmap,
    const uint w,
    const uint h,
    __global int *sc_params,
    __global float *sc_items,
    __global int *tx_params,
    __global uchar *tx_items
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = x + y * w;
    float3 col = RayTracing(w, h, x, y, 1, 
        sc_params, sc_items, tx_params, tx_items);
    col *= 255;
    int res = ((int)col.x << 16) + ((int)col.y << 8) + (int)col.z;
    intmap[pixid] = res;
}

__kernel void clear(
    __global float *floatmap,
    const uint w,
    const uint h
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = (x + y * w) * 3;
    floatmap[pixid + 0] = 0.0f;
    floatmap[pixid + 1] = 0.0f;
    floatmap[pixid + 2] = 0.0f;
}

__kernel void image_from_floatmap(
    __global float *floatmap,
    __global int *imagemap,
    const uint w,
    const uint h,
    const uint AA
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pix_int = (x + y * w);
    uint pix_float = pix_int * 3;
    float aa = (float)aa;
    float r = clamp(floatmap[pix_float + 0], 0.0f, aa) * 255.0f;
    float g = clamp(floatmap[pix_float + 1], 0.0f, aa) * 255.0f;
    float b = clamp(floatmap[pix_float + 2], 0.0f, aa) * 255.0f;
    imagemap[pix_int] = ((int)r << 16) + ((int)g << 8) + (int)b;
}
