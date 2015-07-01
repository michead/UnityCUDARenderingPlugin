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
    private static uint MESH_SIZE = 250;

    private static int UNITY_RENDER_EVENT_ID = 0;
    private static String TEX_NAME_ID = "_MainTex";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyDelegate(string str);

    [DllImport("RenderingPluginDLL")]
    private static extern void UnityRenderEvent(int eventID);
    [DllImport("RenderingPluginDLL")]
    private static extern void Init([In, Out] Vector3[] verts, uint sideSize);
    [DllImport("RenderingPluginDLL")]
    private static extern void Cleanup();
    [DllImport("RenderingPluginDLL")]
    private static extern void SetDebugFunction(IntPtr fp);
    [DllImport("RenderingPluginDLL")]
    private static extern void ComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void ParallelComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void SetTextureFromUnity(System.IntPtr texture);

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

        mesh = new Mesh();
        getMeshDataFromFile(meshURL);
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;

        MyDelegate callbackDelegate = new MyDelegate(callback);
        IntPtr intptrDelegate = Marshal.GetFunctionPointerForDelegate(callbackDelegate);
        SetDebugFunction(intptrDelegate);

        Init(verts, MESH_SIZE);

        Texture2D tex = new Texture2D((int)MESH_SIZE, (int)MESH_SIZE, TextureFormat.RGBAFloat, false);
        tex.LoadRawTextureData(getBytesFromVector3Array(verts));
        tex.Apply();

        // GetComponent<MeshRenderer>().material.SetTexture(TEX_NAME_ID, tex);
        GetComponent<MeshRenderer>().material.mainTexture = tex;
        SetTextureFromUnity(tex.GetNativeTexturePtr());
        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    void OnApplicationQuit()
    {
        Cleanup();
    }

    void computeSineWave()
    {
        float elpasedTime = Time.timeSinceLevelLoad;

        for (int i = 0; i < verts.Length; i++) 
            verts[i].y = (float)(Math.Sin(verts[i].x * FREQ + elpasedTime) * Math.Cos(verts[i].z * FREQ + elpasedTime)) * 0.2f;

        mesh.vertices = verts;
        mesh.RecalculateNormals();
        GetComponent<MeshFilter>().mesh = mesh;
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
        float zero = 0.0f;

        for (int i = 0; i < array.Length; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].x), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].y), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].z), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(zero), 0, bytes, index, 4);
            index += 4;
        }

        return bytes;
    }

    private IEnumerator CallPluginAtEndOfFrames()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            // parallelComputeSineWave();

            GL.IssuePluginEvent(UNITY_RENDER_EVENT_ID);
        }
    }
}