using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;

public class RenderingPlugin : MonoBehaviour
{
    public static Material lineMaterial;
    public static Mesh mesh;

    public static float FREQ = 4.0f;
    public static int SIDE_SIZE = 250;
    public static float SPACING = 0.25f; // totally useless -- DEPRECATED

    struct float3
    {
        public float x, y, z;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyDelegate(string str);

    [DllImport("RenderingPluginDLL")]
    public static extern void SetDebugFunction(IntPtr fp);
    [DllImport("RenderingPluginDLL")]
    private static extern void ComputeSineWave([In, Out] float3[] verts, int sideSize, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void ParallelComputeSineWave([In, Out] float3[] verts, int sideSize, float time);

    public string meshURL;
    Vector3[] verts;
    int[] triangles;

    int numVerts;
    int numFaces;

    void OnPostRender()
    {

    }

    void Start()
    {
        if (meshURL == null)
        {
            Debug.Log("Please choose a mesh to render!");
            return;
        }

        mesh = new Mesh();
        getMeshDataFromFile(meshURL);
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;

        MyDelegate callbackDelegate = new MyDelegate(callback);
        IntPtr intptrDelegate = Marshal.GetFunctionPointerForDelegate(callbackDelegate);
        SetDebugFunction(intptrDelegate);
    }

    void Update()
    {
        // computeSineWave();
        parallelComputeSineWave(SIDE_SIZE, Time.timeSinceLevelLoad);
    }

    # region DEPRECATED
    void generatePlaneMesh()
    {
        generateVerts();
        generateTriangles();

        mesh.vertices = verts;
        mesh.triangles = triangles;
    }

    void generateVerts()
    {
        int index = 0;

        for (int i = -SIDE_SIZE / 2; i < SIDE_SIZE / 2; i++) 
        {
            for (int j = -SIDE_SIZE / 2; j < SIDE_SIZE / 2; j++)
            {
                verts[index] = new Vector3( i * SPACING, 0, j * SPACING );

                index++;
            }
        }
    }

    void generateTriangles()
    {
        int triangleCount = 0;

        for (int i = 0; i < verts.Length; i++)
        {
            if ((i + 1) % SIDE_SIZE != 0)
            {
                if (i + SIDE_SIZE < verts.Length)
                {
                    triangles[triangleCount++] = i + 1;
                    triangles[triangleCount++] = i + SIDE_SIZE;
                    triangles[triangleCount++] = i;
                }
                if (i - SIDE_SIZE > -1)
                {
                    triangles[triangleCount++] = i + 1;
                    triangles[triangleCount++] = i;
                    triangles[triangleCount++] = i - SIDE_SIZE + 1;
                }
            }
        }
    }

    void computeSineWave()
    {
        for (int i = 0; i < verts.Length; i++) 
        {
            float elpasedTime = Time.timeSinceLevelLoad;
            verts[i].y = (float)(Math.Sin(verts[i].x * FREQ + elpasedTime) * Math.Cos(verts[i].z * FREQ + elpasedTime)) * 0.2f;
        }

        mesh.vertices = verts;
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;
    }
    #endregion

    void parallelComputeSineWave(int sideSize, float time) 
    {
        float3[] vertices = convertToFloat3(mesh.vertices);
        ParallelComputeSineWave(vertices, sideSize, time);
        mesh.vertices = convertToVector3(vertices);
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;

        /*
        GCHandle handle = GCHandle.Alloc(convertToFloat(verts), GCHandleType.Pinned);
        try
        {
            IntPtr pointer = handle.AddrOfPinnedObject();
            // ParallelComputeSineWave(ref pointer, sideSize, time);
            ComputeSineWave(ref pointer, sideSize, time);
            
            byte[] data = new byte[numVerts * sizeof(float)];
            float[] fa = new float[numVerts * 3];

            Marshal.Copy(pointer
             , data
             , 0
             , data.Length);
            Buffer.BlockCopy(data, 0, fa, 0, data.Length);

            verts = convertToVector3(fa);
        }
        finally
        {
            if (handle.IsAllocated)
            {
                handle.Free();
            }
        }
        */
    }

    void getMeshDataFromFile(string url)
    {
        getVertsFromFile(url);
        getFacesFromFile(url);

        mesh.vertices = verts;
        mesh.triangles = triangles;
    }

    void getVertsFromFile(string url) 
    {
        numVerts = 0;

        string[] lines = File.ReadAllLines(url);
        foreach(string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                numVerts++;
            }
        }

        int index = 0;

        verts = new Vector3[numVerts];

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                verts[index++] = new Vector3(float.Parse(tokens[1]), float.Parse(tokens[2]), float.Parse(tokens[3]));
            }
        }
    }

    void getFacesFromFile(string url)
    {
        numFaces = 0;

        string[] lines = File.ReadAllLines(url);
        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                numFaces++;
            }
        }

        triangles = new int[numFaces * 3];

        int index = 0;

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                triangles[index++] = int.Parse(tokens[1]) - 1;
                triangles[index++] = int.Parse(tokens[2]) - 1;
                triangles[index++] = int.Parse(tokens[3]) - 1;
            }
        }
    }

    float[] convertToFloat(Vector3[] verts)
    {
        float[] result = new float[verts.Length * 3];
        int index = 0;

        for (int i = 0; i < verts.Length; i++)
        {
            result[index++] = verts[i].x;
            result[index++] = verts[i].y;
            result[index++] = verts[i].z;
        }

        return result;
    }

    float3[] convertToFloat3(Vector3[] verts) 
    {
        float3[] result = new float3[verts.Length];
        for (int i = 0; i < verts.Length; i++)
        {
            Vector3 vec = verts[i];
            result[i] = new float3{ x = vec.x, y = vec.y, z = vec.z};
        }

        return result;
    }

    Vector3[] convertToVector3(float[] verts)
    {
        Vector3[] result = new Vector3[verts.Length / 3];

        for (int i = 0; i < verts.Length; i += 3)
            result[i / 3] = new Vector3(verts[i], verts[i + 1], verts[i + 2]);

        return result;
    }

    Vector3[] convertToVector3(float3[] verts)
    {
        Vector3[] result = new Vector3[verts.Length];
        for (int i = 0; i < verts.Length; i++)
        {
            float3 vec = verts[i];
            result[i] = new Vector3(vec.x, vec.y, vec.z );
        }

        return result;
    }

    static void callback(string str)
    {
        Debug.Log("Callback: " + str);
    }
}