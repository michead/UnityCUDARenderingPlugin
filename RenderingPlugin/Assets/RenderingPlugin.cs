using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;

public class RenderingPlugin : MonoBehaviour
{
    public static Mesh mesh;
    private static Shader shader;
    private static float FREQ = 4.0f;
    private static int MESH_SIZE = 11;
    private static int UNITY_RENDER_EVENT_ID = 0;
    private static Texture2D tex;
    public bool useMeshURL = false;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyDelegate(string str);

    [DllImport("RenderingPluginDLL")]
    private static extern void UnityRenderEvent(int eventID);
    [DllImport("RenderingPluginDLL")]
    private static extern void Init([In, Out] Vector3[] verts, uint sideSize, IntPtr texPtr);
    [DllImport("RenderingPluginDLL")]
    private static extern void Cleanup();
    [DllImport("RenderingPluginDLL")]
    private static extern void SetDebugFunction(IntPtr fp);
    [DllImport("RenderingPluginDLL")]
    private static extern void ComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void ParallelComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void SetTimeFromUnity(float t);

    public string meshURL;
    Vector3[] verts;
    int[] triangles;

    int numVerts;
    int numFaces;

    IEnumerator Start()
    {
        if (meshURL == null)
        {
            Debug.Log("Please choose a mesh to render!");
            yield return null;
        }

        
        if(useMeshURL)
        {
            mesh = new Mesh();
            getMeshDataFromFile(meshURL);
            mesh.RecalculateNormals();
            GetComponent<MeshFilter>().mesh = mesh;
        }
        else
        {
            // For convenience, I'm assuming mesh is a square plane
            MESH_SIZE = (int)Math.Sqrt(GetComponent<MeshFilter>().mesh.vertexCount);
        }

        verts = GetComponent<MeshFilter>().mesh.vertices;

        MyDelegate callbackDelegate = new MyDelegate(callback);
        IntPtr intptrDelegate = Marshal.GetFunctionPointerForDelegate(callbackDelegate);
        SetDebugFunction(intptrDelegate);

        byte[] textureData = getBytesFromVector3Array(verts);
        tex = new Texture2D(MESH_SIZE * 2, MESH_SIZE * 2, TextureFormat.RGBA32, false);
        // FillTextureWithVertices(tex, verts);
        tex.SetPixels32(getColor32ArrayFromVector3Array(verts));
        tex.Apply();
        GetComponent<Renderer>().material.mainTexture = tex;
        // PrintTexture(textureData);

        // Init(verts, (uint)MESH_SIZE, tex.GetNativeTexturePtr());

        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    void Update()
    {
        computeSineWave();
    }

    void OnApplicationQuit()
    {
        // Cleanup();
    }

    void computeSineWave()
    {
        float elpasedTime = Time.timeSinceLevelLoad;

        for (int i = 0; i < verts.Length; i++) 
            verts[i].y = (float)(Math.Sin(verts[i].x * FREQ + elpasedTime) * Math.Cos(verts[i].z * FREQ + elpasedTime)) * 0.2f;

        // mesh.vertices = verts;
        // mesh.RecalculateNormals();
        // GetComponent<MeshFilter>().mesh = mesh;

        tex.SetPixels32(getColor32ArrayFromVector3Array(verts));
        tex.Apply();
    }

    void parallelComputeSineWave() 
    {
        ParallelComputeSineWave(verts, Time.timeSinceLevelLoad);
        mesh.vertices = verts;
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;
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

    static void callback(string str)
    {
        Debug.Log("Callback: " + str);
    }

    byte[] getBytesFromVector3Array(Vector3[] array)
    {
        byte[] bytes = new byte[array.Length * 4 * 4];

        int index = 0;
        float one = 1.0f;

        for (int i = 0; i < array.Length; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].x), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].y), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].z), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(one), 0, bytes, index, 4);
            index += 4;
        }

        return bytes;
    }

    private void PrintTexture(byte[] texData)
    {
        int index = 0;

        for (int i = 0; i < texData.Length; i += 16)
        {
            float x = BitConverter.ToSingle(texData, i);
            float y = BitConverter.ToSingle(texData, i + 4);
            float z = BitConverter.ToSingle(texData, i + 8);
            float w = BitConverter.ToSingle(texData, i + 12);

            Color color = tex.GetPixel(index / MESH_SIZE, index % MESH_SIZE);

            Debug.Log("Color at index " + (i / 16) + ": ( " + color.r + ", " + color.g + ", " + color.b + ", " + color.a + " )");
            Debug.Log("Vertex at index " + (i / 16) + ": ( " + x + ", " + y + ", " + z + ", " + w + " )");
            
            index++;
        }
    }

    private void FillTextureWithVertices(Texture2D tex, Vector3[] vertices)
    {
        for(int i = 0; i < vertices.Length; i++){
            Vector3 v = vertices[i];
            Color color = new Color((v.x + 1) / 2.0f, (v.y + 1) / 2.0f, (v.z + 1) / 2.0f, 1.0f);
            tex.SetPixel(i / MESH_SIZE, i % MESH_SIZE, color);
        }
    }

    private Color32[] getColor32ArrayFromVector3Array(Vector3[] array)
    {
        Color32[] cArray = new Color32[array.Length * 4];
        int index = 0;
        float one = 1.0f;

        for (int i = 0; i < array.Length; i++)
        {
            byte[] x = BitConverter.GetBytes(array[i].x);
            byte[] y = BitConverter.GetBytes(array[i].y);
            byte[] z = BitConverter.GetBytes(array[i].z);
            byte[] w = BitConverter.GetBytes(one);

            cArray[index].r = x[0];
            cArray[index].g = x[1];
            cArray[index].b = x[2];
            cArray[index].a = x[3];

            index++;

            cArray[index].r = y[0];
            cArray[index].g = y[1];
            cArray[index].b = y[2];
            cArray[index].a = y[3];

            index++;

            cArray[index].r = z[0];
            cArray[index].g = z[1];
            cArray[index].b = z[2];
            cArray[index].a = z[3];

            index++;

            cArray[index].r = w[0];
            cArray[index].g = w[1];
            cArray[index].b = w[2];
            cArray[index].a = w[3];

            index++;
        }

        byte[] bytes = new byte[] 
        {
            cArray[0].r,
            cArray[0].g,
            cArray[0].b,
            cArray[0].a,

            cArray[1].r,
            cArray[1].g,
            cArray[1].b,
            cArray[1].a,

            cArray[2].r,
            cArray[2].g,
            cArray[2].b,
            cArray[2].a,

            cArray[3].r,
            cArray[3].g,
            cArray[3].b,
            cArray[3].a
        };

        float xx = BitConverter.ToSingle(bytes, 0);
        float yy = BitConverter.ToSingle(bytes, 4);
        float zz = BitConverter.ToSingle(bytes, 8);
        float ww = BitConverter.ToSingle(bytes, 12);

        // Debug.Log(verts[0].x + "," + xx + " " + verts[0].y + "," + yy + " " + verts[0].z + "," + zz + " " + ww);

        return cArray;
    }

    private IEnumerator CallPluginAtEndOfFrames()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            SetTimeFromUnity(Time.timeSinceLevelLoad);

            // GL.IssuePluginEvent(UNITY_RENDER_EVENT_ID);
        }
    }
}