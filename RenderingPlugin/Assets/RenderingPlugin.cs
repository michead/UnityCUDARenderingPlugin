using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;

public class RenderingPlugin : MonoBehaviour
{
    public static Material lineMaterial;
    public static Mesh mesh;

    public static int WIDTH = 8;
    public static int HEIGHT = 8;
    public static int SPACING = 1;

    public static float FREQ = 4.0f;

    struct float3
    {
        public float x, y, z;
    }

    [DllImport ("RenderingPluginDLL")]
	private static extern void SetTimeFromUnity(float t);
    [DllImport("RenderingPluginDLL")]
    private static extern float ComputeSineWave(float u, float v);
    [DllImport("RenderingPluginDLL")]
    private static extern IntPtr ParallelComputeSineWave(IntPtr verts, float width, float height, float time);

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

        // verts = new Vector3[WIDTH * HEIGHT];
        // triangles = new int[((WIDTH - 1) * (HEIGHT - 1)) * 2 * 3];

        mesh = new Mesh();
        getMeshDataFromFile(meshURL);
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;

        // generatePlaneMesh();
    }

    void Update()
    {
        // SetTimeFromUnity(Time.timeSinceLevelLoad);
        // computeSineWave();
        parallelComputeSineWave(WIDTH, HEIGHT, Time.timeSinceLevelLoad);
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

        for (int i = -WIDTH / 2; i < WIDTH / 2; i++) 
        {
            for (int j = -HEIGHT / 2; j < HEIGHT / 2; j++)
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
            if((i + 1) % WIDTH != 0)
            {
                if (i + WIDTH < verts.Length)
                {
                    triangles[triangleCount++] = i + 1;
                    triangles[triangleCount++] = i + WIDTH;
                    triangles[triangleCount++] = i;
                }
                if(i - WIDTH > -1)
                {
                    triangles[triangleCount++] = i + 1;
                    triangles[triangleCount++] = i;
                    triangles[triangleCount++] = i - WIDTH + 1;
                }
            }
        }
    }

    void computeSineWave()
    {
        for (int i = 0; i < verts.Length; i++) 
        {
            verts[i].y = ComputeSineWave(i % WIDTH, i / WIDTH);
        }

        mesh.vertices = verts;
        mesh.RecalculateNormals();
    }
    #endregion

    void parallelComputeSineWave(float width, float height, float time) 
    {
        GCHandle handle = GCHandle.Alloc(verts, GCHandleType.Pinned);
        try
        {
            IntPtr pointer = handle.AddrOfPinnedObject();
            IntPtr res = ParallelComputeSineWave(pointer, width, height, time);
            
            byte[] data = new byte[numVerts * sizeof(float)];
            Marshal.Copy(pointer
             , data
             , 0
             , data.Length);
            Buffer.BlockCopy(data, 0, verts, 0, data.Length);
        }
        finally
        {
            if (handle.IsAllocated)
            {
                handle.Free();
            }
        }
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
}