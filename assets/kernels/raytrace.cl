#define MAX_RENDER_DIST 1000000.0f
#define MAX_RENDER_DEPTH 4
#define EPSILON 0.0001f
#define PI4 12.5663f
#define PI2 6.28317f
#define PI 3.141592f
#define INV_PI 0.3183098f
#define AMBIENT 0.05f
#define GAMMA 2.2f

struct Material{
    float3 col;
    float reflectivity;
    float3 absorption;
    float refraction;
    float roughness;
    float emittance;
    uint texture;
    uint normalmap;
    uint roughnessmap;
    uint metalicmap;
    float texscale;
};

#define pPLANE 6
#define pSPHERE 4
#define pTRI 9

struct RayHit{
    float3 pos;
    float3 nor;
    float t;
    uint mat_index;
    uchar ptype;
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
// sizes, must be the same as rust provides
#define SC_MAT_SIZE 15
#define SC_LIGHT_SIZE 7
#define SC_PLANE_SIZE 7
#define SC_SPHERE_SIZE 5
#define SC_TRI_SIZE 10

struct Scene{
    uint *params, *tex_params;
    float *items;
    uchar *textures;
    uint *bvh;
    uint skybox;
    float3 skycol;
    float skyintens;
};

//first byte in array where this type starts
uint ScGetStart(uint type, struct Scene *scene){
    return scene->params[type * 2 + 1];
}

//number of items of this type(not bytes!)
uint ScGetCount(uint type, struct Scene *scene){
    return scene->params[type * 2];
}

//extract material from array, off is index of first byte of material we want
struct Material ExtractMaterial(uint off, float *arr){
    struct Material mat;
    mat.col = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
    mat.reflectivity = arr[off + 3];
    mat.absorption = (float3)(arr[off + 4], arr[off + 5], arr[off + 6]);
    mat.refraction = arr[off + 7];
    mat.roughness = arr[off + 8] + EPSILON;
    mat.emittance = arr[off + 9];
    mat.texture = (uint)arr[off + 10];
    mat.normalmap = (uint)arr[off + 11];
    mat.roughnessmap = (uint)arr[off + 12];
    mat.metalicmap  = (uint)arr[off + 13];
    mat.texscale = arr[off + 14];
    return mat;
}

struct Material GetMaterialFromIndex(uint index, struct Scene *scene){
    uint start = ScGetStart(SC_MAT, scene);
    return ExtractMaterial(start + index * SC_MAT_SIZE, scene->items);
}

//first byte of texture
uint TxGetStart(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 0];
}

uint TxGetWidth(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 1];
}

uint TxGetHeight(uint tex, struct Scene *scene){
    return scene->tex_params[tex * 3 + 2];
}

//get sample
float3 TxGetSample(uint tex, struct Scene *scene, uint x, uint y, uint w){
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
float3 GetTexCol(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    return pow(TxGetSample(tex, scene, x, y, w), GAMMA);
}

//get value to range 0..1 (no gamma)
float3 GetTexVal(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    return TxGetSample(tex, scene, x, y, w);
}

//get value 0..1 from scalar map
float GetTexScalar(uint tex, float2 uv, struct Scene *scene){
    UV_TO_XY;
    uint offset = TxGetStart(tex, scene) + (y * w + x);
    float scalar = (float)scene->textures[offset];
    return scalar / 255.0f;
}

//Copy a float3 out the array, off(offset) is the first byte of the float3 we want
float3 ExtractFloat3(uint off, float *arr){
    return (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
}

float3 ExtractFloat3FromInts(uint *arr, uint index){
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
    float d = tca * tca - dot(l, l) + srad * srad;
    if(d < 0.0f) return false;
    float t = tca - sqrt(d);
    if(t < 0.0f){
        t = tca + sqrt(d);
        if (t < 0.0f) return false;
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
// https://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
float InterAABB(float3 rpos, float3 rdinv, float3 bmin, float3 bmax){
    float3 a = (bmin - rpos) * rdinv;
    float3 b = (bmax - rpos) * rdinv;
    float3 temp = fmin(a, b);
    float tmin = fmax(fmax(temp.x, temp.y), temp.z);
    temp = fmax(a, b);
    float tmax = fmin(fmin(temp.x, temp.y), temp.z);
    if(tmax < 0.0f || tmin > tmax){
        return MAX_RENDER_DIST;
    }
    return tmin;
}

//plane uv
float2 PlaneUV(float3 pos, float3 nor){
    float3 u = (float3)(nor.y, nor.z, -nor.x);
    float3 v = fast_normalize(cross(u, nor));
    return (float2)(dot(pos, u),dot(pos, v));
}

//sphere uv
float2 SphereUV(float3 nor){
    float u = 0.5f + (atan2(-nor.z, -nor.x) / (2 * M_PI));
    float v = 0.5f - asinpi(-nor.y);
    return (float2)(u, v);
}

//sphere skybox uv(just sphere uv with inverted normal)
float2 SkySphereUV(float3 nor){
    float u = 0.5f + (atan2(nor.z, nor.x) / (2 * M_PI));
    float v = 0.5f - asinpi(nor.y);
    return (float2)(u, v);
}

//macros for primitive intersections
#define START_PRIM() \
    (struct RayHit *closest, struct Ray *ray, float *arr, const uint count, const uint start, const uint stride){\
    uint coff = UINT_MAX;\
    for(uint i = 0; i < count; i++){\
        uint off = start + i * stride;\

#define END_PRIM(ptyp) {\
            if(hit) coff = off;\
        }\
        if(coff != UINT_MAX){\
            closest->mat_index = arr[coff + ptyp];\
            closest->ptype = ptyp;\
        }\
    }\
}\

//actual functions
void InterSpheres START_PRIM()
    float3 spos = ExtractFloat3(off + 0, arr);
    float srad = arr[off + 3];
    bool hit = InterSphere(ray, closest, spos, srad);
END_PRIM(pSPHERE)

void InterPlanes START_PRIM()
    float3 ppos = ExtractFloat3(off + 0, arr);
    float3 pnor = ExtractFloat3(off + 3, arr);
    bool hit = InterPlane(ray, closest, ppos, pnor);
END_PRIM(pPLANE)

void InterTris START_PRIM()
    float3 a = ExtractFloat3(off + 0, arr);
    float3 b = ExtractFloat3(off + 3, arr);
    float3 c = ExtractFloat3(off + 6, arr);
    bool hit = InterTri(ray, closest, a, b, c);
END_PRIM(pTRI)

//intersect whole scene
struct RayHit InterScene(struct Ray *ray, struct Scene *scene){
    struct RayHit closest = NullRayHit();
    InterPlanes(&closest, ray, scene->items, ScGetCount(SC_PLANE, scene), ScGetStart(SC_PLANE, scene), SC_PLANE_SIZE);
    InterSpheres(&closest, ray, scene->items, ScGetCount(SC_SPHERE, scene), ScGetStart(SC_SPHERE, scene), SC_SPHERE_SIZE);
    InterTris(&closest, ray, scene->items, ScGetCount(SC_TRI, scene), ScGetStart(SC_TRI, scene), SC_TRI_SIZE);
    return closest;
}

float3 Yawed(float3 v, float o){
    float coso = cos(o);
    float sino = sin(o);
    return (float3)(
        v.x * coso + v.z * sino,
        v.y,
        -v.x * sino + v.z * coso
    );
}

// (yaw, roll)
float2 Orientation(float3 v){
    float3 w = fast_normalize(v);
    return (float2)(
        atan2(w.x, -w.z),
        asin(w.y)
    );
}

// adapted from our rust version and nvidia dev blog:
// https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
struct RayHit InterSceneBvh(struct Ray *inray, struct Scene *scene){
    struct Ray ray = *inray;
    struct RayHit closest = NullRayHit();
    float3 rdinv = 1.0f / ray.dir;

    uint primitives_start = scene->bvh[0];
    uint models_start = scene->bvh[1];
    uint vertices_start = scene->bvh[2];

    uint sph_start = ScGetStart(SC_SPHERE, scene);
    uint sph_count = ScGetCount(SC_SPHERE, scene);
    uint tri_start = ScGetStart(SC_TRI, scene);
    uint pla_count = ScGetCount(SC_PLANE, scene);
    uint pla_start = ScGetStart(SC_PLANE, scene);

    uchar ptype = 0;
    uint coff = UINT_MAX;

    // Planes are not in the bvh so we just check em linearly
    for(uint i = 0; i < pla_count; i++){
        uint off = pla_start + i * SC_PLANE_SIZE;
        float3 ppos = ExtractFloat3(off + 0, scene->items);
        float3 pnor = ExtractFloat3(off + 3, scene->items);
        bool hit = InterPlane(&ray, &closest, ppos, pnor);
        if(hit){
            coff = off;
            ptype = pPLANE;
        }
    }

    #define size 32
    uint stack[size];
    stack[0] = UINT_MAX;
    stack[1] = 0;
    uint ptr = 2;
    bool toplevel = true;
    uint first_toplevel_vertex = UINT_MAX;
    uint mesh_start = 0;
    struct Ray backup_ray;
    float3 backup_rdinv;
    float model_yaw;
    float3 model_pos;
    bool hit_in_mesh = false;

    while(ptr < size){
        uint current = stack[--ptr];
        if(current == UINT_MAX) break;
        if(ptr == first_toplevel_vertex){
            toplevel = true;
            vertices_start = scene->bvh[2];
            ray = backup_ray;
            rdinv = backup_rdinv;
            if(hit_in_mesh){
                closest.nor = Yawed(closest.nor, model_yaw);
                closest.pos = Yawed(closest.pos, model_yaw);
                closest.pos += model_pos;
            }
            hit_in_mesh = false;
        }
        uint v = vertices_start + current * 8;
        uint left_first = scene->bvh[v + 6];
        uint count = scene->bvh[v + 7];

        if(count > 0){ // leaf
            if(toplevel){ // handle top level primitive leaf or start the traversal of the mesh bvh
                for(uint i = left_first; i < left_first + count; i++){
                    uint prim_type = scene->bvh[primitives_start + (i * 2) + 0];
                    uint prim_index = scene->bvh[primitives_start + (i * 2) + 1];
                    // intersect primitive
                    if(prim_type == 0){ // model
                        toplevel = false;
                        first_toplevel_vertex = ptr - 1;
                        uint model_start = models_start + (prim_index * 8);
                        model_pos = ExtractFloat3FromInts(scene->bvh, model_start + 0);
                        float3 rot = ExtractFloat3FromInts(scene->bvh, model_start + 3);
                        uint mat = scene->bvh[model_start + 6];
                        uint mesh = scene->bvh[model_start + 7];
                        vertices_start = scene->bvh[(mesh + 1) * 2 + 2];
                        mesh_start = scene->bvh[(mesh + 1) * 2 + 3];
                        stack[ptr++] = 0;
                        model_yaw = Orientation(rot).x;
                        backup_ray = ray;
                        backup_rdinv = rdinv;
                        ray.pos = ray.pos - model_pos;
                        ray.pos = Yawed(ray.pos, -model_yaw);
                        ray.dir = Yawed(ray.dir, -model_yaw);
                        rdinv = 1.0f / ray.dir;
                    } else if(prim_type == 1){ // sphere
                        uint off = sph_start + prim_index * SC_SPHERE_SIZE;
                        float3 spos = ExtractFloat3(off + 0, scene->items);
                        float srad = scene->items[off + 3];
                        bool hit = InterSphere(&ray, &closest, spos, srad);
                        if(hit){
                            coff = off;
                            ptype = pSPHERE;
                        }
                    } else { // triangle
                        uint off = tri_start + prim_index * SC_TRI_SIZE;
                        float3 a = ExtractFloat3(off + 0, scene->items);
                        float3 b = ExtractFloat3(off + 3, scene->items);
                        float3 c = ExtractFloat3(off + 6, scene->items);
                        bool hit = InterTri(&ray, &closest, a, b, c);
                        if(hit){
                            coff = off;
                            ptype = pTRI;
                        }
                    }
                }
            } else { // handle triangles in leaf of mesh bvh
                for(uint i = left_first; i < left_first + count; i++){
                    uint off = mesh_start + i * SC_TRI_SIZE;
                    float3 a = ExtractFloat3(off + 0, scene->items);
                    float3 b = ExtractFloat3(off + 3, scene->items);
                    float3 c = ExtractFloat3(off + 6, scene->items);
                    bool hit = InterTri(&ray, &closest, a, b, c);
                    if(hit){
                        // hit.mat = model.mat;
                        coff = off;
                        ptype = pTRI;
                        hit_in_mesh = true;
                    }
                }
            }
        } else { // traverse tree, either top level or mesh level (all the same)
            uint vertices[2] = {
                left_first,
                left_first + 1,
            };

            v = vertices_start + left_first * 8;
            float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
            float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
            float t0 = InterAABB(ray.pos, rdinv, bmin, bmax);

            v = vertices_start + (left_first + 1) * 8;
            bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
            bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
            float t1 = InterAABB(ray.pos, rdinv, bmin, bmax);
            float ts[2] = { t0, t1 };

            uint order[2] = { 0, 0 };
            if(ts[0] <= ts[1]){
                order[1] = 1;
            } else {
                order[0] = 1;
            }

            if(ts[order[0]] < closest.t){
                stack[ptr++] = vertices[order[0]];
                if(ts[order[1]] < closest.t){
                    stack[ptr++] = vertices[order[1]];
                }
            }
            else if(ts[order[1]] < closest.t){
                stack[ptr++] = vertices[order[1]];
            }
        }
    }

    if(coff != UINT_MAX){
        closest.mat_index = scene->items[coff + ptype];
        closest.ptype = ptype;
    }

    return closest;
}

uint InterTest(struct Ray *ray, struct Scene *scene){
    uint test = 0;
    struct RayHit closest = NullRayHit();
    float3 rpos = ray->pos;
    float3 rdinv = 1.0f / ray->dir;

    uint primitive_start = scene->bvh[0];
    uint vertex_start = scene->bvh[1];

    uint sph_start = ScGetStart(SC_SPHERE, scene);
    uint sph_count = ScGetCount(SC_SPHERE, scene);
    uint tri_start = ScGetStart(SC_TRI, scene);
    uint pla_count = ScGetCount(SC_PLANE, scene);
    uint pla_start = ScGetStart(SC_PLANE, scene);

    uchar ptype = 0;
    uint coff = UINT_MAX;

    // Planes are not in the bvh so we just check em linearly
    for(uint i = 0; i < pla_count; i++){
        uint off = pla_start + i * SC_PLANE_SIZE;
        float3 ppos = ExtractFloat3(off + 0, scene->items);
        float3 pnor = ExtractFloat3(off + 3, scene->items);
        bool hit = InterPlane(ray, &closest, ppos, pnor);
        if(hit){
            coff = off;
            ptype = pPLANE;
        }
    }

    #define size 64
    uint stack[size];
    stack[0] = UINT_MAX;
    stack[1] = 0;
    uint ptr = 2;

    while(ptr < size){
        uint current = stack[--ptr];
        if(current == UINT_MAX) break;
        uint v = vertex_start + current * 8;
        uint left_first = scene->bvh[v + 6];
        uint count = scene->bvh[v + 7];

        if(count > 0){ // leaf
            for(uint i = left_first; i < left_first + count; i++){
                uint prim_type = scene->bvh[primitive_start + (i * 2) + 0];
                uint prim_index = scene->bvh[primitive_start + (i * 2) + 1];
                // intersect primitive
                if(prim_type == 0){ // model
                    continue; // TODO
                } else if(prim_type == 1){ // sphere
                    uint off = sph_start + prim_index * SC_SPHERE_SIZE;
                    float3 spos = ExtractFloat3(off + 0, scene->items);
                    float srad = scene->items[off + 3];
                    bool hit = InterSphere(ray, &closest, spos, srad);
                    if(hit){
                        coff = off;
                        ptype = pSPHERE;
                    }
                } else { // triangle
                    uint off = tri_start + prim_index * SC_TRI_SIZE;
                    float3 a = ExtractFloat3(off + 0, scene->items);
                    float3 b = ExtractFloat3(off + 3, scene->items);
                    float3 c = ExtractFloat3(off + 6, scene->items);
                    bool hit = InterTri(ray, &closest, a, b, c);
                    if(hit){
                        coff = off;
                        ptype = pTRI;
                    }
                }
            }
        } else {
            uint vertices[2] = {
                left_first,
                left_first + 1,
            };

            v = vertex_start + left_first * 8;
            float3 bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
            float3 bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
            float t0 = InterAABB(rpos, rdinv, bmin, bmax);

            v = vertex_start + (left_first + 1) * 8;
            bmin = ExtractFloat3FromInts(scene->bvh, v + 0);
            bmax = ExtractFloat3FromInts(scene->bvh, v + 3);
            float t1 = InterAABB(rpos, rdinv, bmin, bmax);
            float ts[2] = { t0, t1 };

            uint order[2] = { 0, 0 };
            if(ts[0] <= ts[1]){
                order[1] = 1;
            } else {
                order[0] = 1;
            }

            if(/*ts[order[0]] >= 0.0f && */ts[order[0]] < closest.t){
                stack[ptr++] = vertices[order[0]];
                test++;
                if(/*ts[order[1]] >= 0.0f && */ts[order[1]] < closest.t){
                    stack[ptr++] = vertices[order[1]];
                    test++;
                }
            }
            else if(/*ts[order[1]] >= 0.0f && */ts[order[1]] < closest.t){
                stack[ptr++] = vertices[order[1]];
                test++;
            }
        }
    }

    if(coff != UINT_MAX){
        closest.mat_index = scene->items[coff + ptype];
        closest.ptype = ptype;
    }

    return test;
}

// #define INTER_SCENE InterScene
#define INTER_SCENE InterSceneBvh

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
    float specangle = max(dot(halfdir, nor), 0.0f);
    float spec = pow(specangle, 16.0f / roughness);
    return power * (float2)(angle, spec);
}

//get diffuse light incl colour of hit with all lights
void Blinn(struct RayHit *hit, struct Scene *scene, float3 viewdir, float3 colour, float roughness, float3 *out_diff, float3 *out_spec){
    float3 col = (float3)(AMBIENT);
    float3 spec = (float3)(0.0f);
    float* arr = scene->items;
    uint count = ScGetCount(SC_LIGHT, scene);
    uint start = ScGetStart(SC_LIGHT, scene);
    for(uint i = 0; i < count; i++){
        uint off = start + i * SC_LIGHT_SIZE;
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

#define HANDLE_TEXTURES\
    /*texture*/\
    float2 uv;\
    if(mat.texture > 0){\
        uchar ptype = hit.ptype;\
        if(ptype == pPLANE || ptype == pTRI)\
            uv = PlaneUV(hit.pos, hit.nor);\
        else if(ptype == pSPHERE)\
            uv = SphereUV(hit.nor);\
        uv *= mat.texscale;\
        mat.col *= GetTexCol(mat.texture - 1, uv, scene);\
    }\
    /*normalmap*/\
    if(mat.normalmap > 0){\
        float3 rawnor = GetTexVal(mat.normalmap - 1, uv, scene);\
        float3 t = cross(hit.nor, (float3)(0.0f,1.0f,0.0f));\
        if(fast_length(t) < EPSILON)\
            t = cross(hit.nor, (float3)(0.0f,0.0f,1.0f));\
        t = fast_normalize(t);\
        float3 b = fast_normalize(cross(hit.nor, t));\
        rawnor = rawnor * 2 - 1;\
        rawnor = fast_normalize(rawnor);\
        float3 newnor;\
        float3 row = (float3)(t.x, b.x, hit.nor.x);\
        newnor.x = dot(row, rawnor);\
        row = (float3)(t.y, b.y, hit.nor.y);\
        newnor.y = dot(row, rawnor);\
        row = (float3)(t.z, b.z, hit.nor.z);\
        newnor.z = dot(row, rawnor);\
        hit.nor = fast_normalize(newnor);\
    }\
    /*roughnessmap*/\
    if(mat.roughnessmap > 0){\
        float value = GetTexScalar(mat.roughnessmap - 1, uv, scene);\
        mat.roughness = value * mat.roughness;\
    }\
    /*metalicmap*/\
    if(mat.metalicmap > 0){\
        float value = GetTexScalar(mat.metalicmap - 1, uv, scene);\
        mat.reflectivity *= value;\
    }\

//Recursion only works with one function
float3 RayTrace(struct Ray *ray, struct Scene *scene, uint depth){
    if(depth == 0) return SkyCol(ray->dir, scene);

    //hit
    struct RayHit hit = INTER_SCENE(ray, scene);
    if(hit.t >= MAX_RENDER_DIST)
        return SkyCol(ray->dir, scene);
    struct Material mat = GetMaterialFromIndex(hit.mat_index, scene);

    HANDLE_TEXTURES;

    //diffuse, specular
    float3 diff, spec;
    Blinn(&hit, scene, ray->dir, mat.col, mat.roughness, &diff, &spec);

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

// -----------------------------------------

float D_GGX(float dnh, float alpha2){
    float dnh2 = pow(dnh, 2.0f);
    return alpha2 / (PI * pow(dnh2 * (alpha2 - 1.0f) + 1.0f, 2.0f));
}

float G_GGX_Smith(float dnw, float alpha2){
    return 2.0f * dnw / (dnw + sqrt(alpha2 + (1.0f - alpha2) * pow(dnw, 2.0f)));
}

float3 Schlick(float dih, float3 kSpecular){
    return kSpecular + ((float3)(1.0f) - kSpecular) * pow(1.0f - max(0.0f, dih), 5.0f);
}

// takes vectors: to light, -view, normal
float3 MicroFacetBRDF(float3 wo, float3 wi, float3 n, float3 kSpecular, float alpha){
    float3 wh = fast_normalize(wo + wi);
    float alpha2 = alpha * alpha;
    float dwin = clamp(dot(wi, n), EPSILON, 1.0f);
    float dwon = clamp(dot(wo, n), EPSILON, 1.0f);
    float dwhn = clamp(dot(wh, n), EPSILON, 1.0f);

    float3 F = Schlick(dot(wi, wh), kSpecular);
    float G = G_GGX_Smith(dwin, alpha2) * G_GGX_Smith(dwon, alpha2);
    float D = D_GGX(dwhn, alpha2);

    return (F * G * D) / (4.0f * dwin * dwon);
}

// -----------------------------------------

// credit: George Marsaglia
uint Xor32(uint* seed){
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

float U32tf01(uint i){
   return (float)i * 2.3283064e-10;
}

// Wang hash
// https://riptutorial.com/opencl/example/20715/using-thomas-wang-s-integer-hash-function
uint WangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Sphere sampling adapted from University of Mons lecture slides
// https://angms.science/doc/RM/randUnitVec.pdf
float3 RandomSpherePoint(uint* seed){
    float a = U32tf01(Xor32(seed)) * PI2;
    float b = U32tf01(Xor32(seed)) * PI2;
    float z = cos(b);
    float z2 = z * z;
    return (float3)(sqrt(1.0f - z2) * cos(a), sqrt(1.0f - z2) * sin(a), z);
}

// https://www.shadertoy.com/view/4s3cRr
// Align along normal method taken from a shadertoy
float3 RandomHemispherePoint(uint* seed, float3 normal){
    float3 dir = RandomSpherePoint(seed);
    return dot(dir, normal) < 0.0 ? -dir : dir;
}

float3 PathTrace(struct Ray ray, struct Scene *scene, uint* seed){
    //return ((float)InterTest(&ray, scene) / 32.0f) * (float3)(1.0f);

    float3 E = (float3)(1.0f); // emittance accumulator
    float ncontext = 1.0f; // refraction index of current medium
    uint tirs = 0; // total internal reflections
    float3 hitpos = ray.pos;
    float3 distacc = (float3)(1.0f);
    while(tirs < 8){
        struct RayHit hit = INTER_SCENE(&ray, scene);
        if(hit.t >= MAX_RENDER_DIST){
            E *= SkyCol(ray.dir, scene) * 3.0f;
            break;
        }

        struct Material mat = GetMaterialFromIndex(hit.mat_index, scene);
        if(mat.emittance > EPSILON){
            E *= mat.col * mat.emittance;
            break;
        }

        struct Ray nray;
        nray.pos = hit.pos + hit.nor * EPSILON;

        // handle dielectrics
        float mf = mat.refraction;
        if(mf > EPSILON){
            bool outside = dot(hit.nor, ray.dir) < 0.0;
            float n1, n2;
            if(outside){
                n2 = mf;
                n1 = ncontext;
            } else {
                // do we have absorption that we should handle?
                if(dot(mat.absorption, mat.absorption) > EPSILON){
                    float dist = fast_length(hitpos - hit.pos);
                    E *= exp(mat.absorption * dist);
                }
                hit.nor *= -1.0f;
                n2 = ncontext;
                n1 = mf;
            }
            hitpos = hit.pos;
            float n = n1 / n2;
            float cost1 = dot(hit.nor, -ray.dir);
            float k = 1.0f - n * n * (1.0f - cost1 * cost1);
            float costt = sqrt(k);
            float3 refldir = reflect(ray.dir, hit.nor);
            if(k < 0.0f){ // total internal reflection
                nray.dir = refldir;
                nray.pos = hit.pos + hit.nor * EPSILON;
                ray = nray;
                tirs++;
                continue;
            }

            float spol = (n1 * cost1 - n2 * costt) / (n1 * cost1 + n2 * costt);
            float ppol = (n1 * costt - n2 * cost1) / (n1 * costt + n2 * cost1);
            float fr = 0.5f * (spol * spol + ppol * ppol);

            // choose reflect or refract
            float decider = U32tf01(Xor32(seed));
            if(decider <= fr){
                nray.dir = refldir;
                nray.pos = hit.pos + hit.nor * EPSILON;
            }else{
                nray.dir = fast_normalize((ray.dir - hit.nor * -cost1) * n + hit.nor * -sqrt(k));
                nray.pos = hit.pos - hit.nor * EPSILON;
                ncontext = mf;
            }

            ray = nray;
            continue;
        }

        // handle non-dielectrics: blend of specular and diffuse
        // HANDLE_TEXTURES;

        // if(mat.reflectivity > EPSILON){ // mirror
        //     float decider = U32tf01(Xor32(seed));
        //     if(decider <= mat.reflectivity){
        //         nray.dir = reflect(ray.dir, hit.nor);
        //         E *= mat.col;
        //         ray = nray;
        //         continue;
        //     }
        // }
        nray.dir = RandomHemispherePoint(seed, hit.nor);
        // float3 BRDF = mat.col * INV_PI;
        float3 BRDF = MicroFacetBRDF(nray.dir, -ray.dir, hit.nor, (float3)(0.95, 0.64, 0.54), mat.roughness * 0.3f);
        float INV_PDF = PI2; // PDF = 1 / 2PI
        float3 Ei = max(dot(hit.nor, nray.dir), 0.0f) * INV_PDF;

        E *= BRDF * Ei;
        ray = nray;
    }
    return E;
}

#define SETUP_SCENE\
    struct Scene scene;\
    scene.params = sc_params;\
    scene.items = sc_items;\
    scene.tex_params = tx_params;\
    scene.textures = tx_items;\
    scene.bvh = bvh;\
    scene.skybox = sc_params[2 * SC_SCENE + 0];\
    scene.skycol = ExtractFloat3FromInts(sc_params, 2 * SC_SCENE + 1);\
    scene.skyintens = as_float(sc_params[2 * SC_SCENE + 4]);\

#define CREATE_RAY(uv)\
    struct Ray ray;\
    ray.pos = ExtractFloat3FromInts(sc_params, 2 * SC_SCENE + 5);\
    float3 cd = fast_normalize(ExtractFloat3FromInts(sc_params, 2 * SC_SCENE + 8));\
    float3 hor = fast_normalize(cross(cd, (float3)(0.0f, 1.0f, 0.0f)));\
    float3 ver = fast_normalize(cross(hor, cd));\
    float3 to = ray.pos + cd;\
    to += uv.x * hor;\
    to += uv.y * ver;\
    ray.dir = fast_normalize(to - ray.pos);\

float3 RayTracing(const uint w, const uint h, const uint x, const uint y,
    __global uint *sc_params, __global float *sc_items, __global uint *tx_params,
    __global uchar *tx_items, __global uint *bvh
){
    SETUP_SCENE;
    float2 uv = (float2)((float)x / w, (float)y / h);
    uv -= 0.5f;
    uv *= (float2)((float)w / h, -1.0f);
    CREATE_RAY(uv);

    float3 col = RayTrace(&ray, &scene, MAX_RENDER_DEPTH);
    col = pow(col, (float3)(1.0f / GAMMA));
    col = clamp(col, 0.0f, 1.0f);
    return col;
}

float3 PathTracing(const uint w, const uint h, const uint x, const uint y, const uint rseed,
    __global uint *sc_params, __global float *sc_items, __global uint *tx_params,
    __global uchar *tx_items, __global uint *bvh
){
    SETUP_SCENE;

    uint hash = WangHash(rseed);
    float u = U32tf01(Xor32(&hash));
    float v = U32tf01(Xor32(&hash));
    float2 uv = (float2)(((float)x + u) / w, ((float)y + v) / h);
    uv -= 0.5f;
    uv *= (float2)((float)w / h, -1.0f);
    CREATE_RAY(uv);

    float3 col = PathTrace(ray, &scene, &hash);
    // col = pow(col, (float3)(1.0f / GAMMA));
    return col;
}

float3 AcesTonemap(float3 x){
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

float3 HablePartial(float3 x){
    float a = 0.15f;
    float b = 0.50f;
    float c = 0.10f;
    float d = 0.20f;
    float e = 0.02f;
    float f = 0.30f;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

float3 HableTonemap(float3 x){
    float exposure_bias = 4.0f; // was 2, but this seems to match aces brightness better
    float3 curr = HablePartial(x * exposure_bias);
    float3 w = (float3)(11.2f);
    float3 white_scale = (float3)(1.0f) / HablePartial(w);
    return clamp(curr * white_scale, 0.0f, 1.0f);
}

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
    float3 col = RayTracing(w, h, x, y,
        sc_params, sc_items, tx_params, tx_items, bvh);
    col *= 255;
    uint res = ((uint)col.x << 16) + ((uint)col.y << 8) + (uint)col.z;
    intmap[pixid] = res;
}

__kernel void pathtracing(
    __global float *floatmap,
    const uint w,
    const uint h,
    const uint t,
    __global uint *sc_params,
    __global float *sc_items,
    __global uint *bvh,
    __global uint *tx_params,
    __global uchar *tx_items
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = x + y * w;
    float3 col = PathTracing(w, h, x, y, pixid * (t + 1),
        sc_params, sc_items, tx_params, tx_items, bvh);
    uint fid = pixid * 3;
    floatmap[fid + 0] += col.x;
    floatmap[fid + 1] += col.y;
    floatmap[fid + 2] += col.z;
}

__kernel void clear(
    __global float *floatmap,
    const uint w
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
    const float mult
){
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixid = (x + y * w);
    uint pix_float = pixid * 3;
    float3 c = (float3)(
                floatmap[pix_float + 0] * mult,
                floatmap[pix_float + 1] * mult,
                floatmap[pix_float + 2] * mult
    );
    //c = AcesTonemap(c);
    c = HableTonemap(c);
    c = pow(c, (float3)(1.0f / GAMMA));
    c *= 255.0f;
    imagemap[pixid] = ((uint)((uchar)c.r) << 16) + ((uint)((uchar)c.g) << 8) + (uint)((uchar)c.b);
}
