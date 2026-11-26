using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// プロセスが利用可能な物理コア数の検出。SMT による FMA/キャッシュ競合を避けるため、
// 既定スレッド数は物理コア数とし、取得不能時は Environment.ProcessorCount にフォールバックする。
internal static class CpuTopology
{
    internal static int PhysicalCores { get; } = Detect();

    private static int Detect()
    {
        int logical = Environment.ProcessorCount;
        int cores = 0;
        try
        {
            if (OperatingSystem.IsWindows())
            {
                cores = WindowsCores();
            }
            else if (OperatingSystem.IsLinux())
            {
                cores = LinuxCores();
            }
            else if (OperatingSystem.IsMacOS())
            {
                cores = MacCores();
            }
        }
        catch (Exception)
        {
            // 取得できない環境（権限・サンドボックスなど）では論理プロセッサ数に戻す
            cores = 0;
        }

        // ProcessorCount は affinity やコンテナの CPU 制限を反映しているので、それを超えない
        return cores > 0 ? Math.Min(cores, logical) : logical;
    }

    // プロセスの CPU アフィニティマスク（64 論理プロセッサ以下で取得可能な場合のみ）。
    private static ulong? Affinity()
    {
        if (Environment.ProcessorCount > 64 || !(OperatingSystem.IsWindows() || OperatingSystem.IsLinux()))
        {
            return null;
        }

        using var process = Process.GetCurrentProcess();
        return (ulong)(long)process.ProcessorAffinity;
    }

    private static int LinuxCores()
    {
        ulong? affinity = Affinity();
        var seen = new HashSet<(string Package, string Core)>();
        foreach (string dir in Directory.EnumerateDirectories("/sys/devices/system/cpu", "cpu*"))
        {
            string name = Path.GetFileName(dir);
            if (!int.TryParse(name.AsSpan(3), out int cpu))
            {
                continue;
            }

            if (affinity is { } mask && (cpu >= 64 || (mask & (1UL << cpu)) == 0))
            {
                continue;
            }

            string coreFile = Path.Combine(dir, "topology", "core_id");
            if (!File.Exists(coreFile))
            {
                continue;
            }

            string packageFile = Path.Combine(dir, "topology", "physical_package_id");
            string package = File.Exists(packageFile) ? File.ReadAllText(packageFile).Trim() : "0";
            seen.Add((package, File.ReadAllText(coreFile).Trim()));
        }

        return seen.Count;
    }

    private static int MacCores()
    {
        int value = 0;
        nint size = sizeof(int);
        return SysctlByName("hw.physicalcpu", ref value, ref size, 0, 0) == 0 ? value : 0;
    }

    private static unsafe int WindowsCores()
    {
        const int RelationProcessorCore = 0;
        uint length = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, null, ref length);
        if (length == 0)
        {
            return 0;
        }

        byte[] buffer = new byte[length];
        fixed (byte* p = buffer)
        {
            if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, ref length))
            {
                return 0;
            }

            // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship(4) Size(4) PROCESSOR_RELATIONSHIP{ Flags(1) EfficiencyClass(1)
            // Reserved(20) GroupCount(2) GroupMask[GroupCount] }。GROUP_AFFINITY は { KAFFINITY Mask, WORD Group, WORD Reserved[3] }
            ulong? affinity = Affinity();
            int maskSize = IntPtr.Size;
            int groupSize = maskSize + 8;
            int cores = 0;
            for (uint offset = 0; offset + 8 <= length;)
            {
                byte* record = p + offset;
                uint size = *(uint*)(record + 4);
                if (size == 0)
                {
                    break;
                }

                if (*(int*)record == RelationProcessorCore)
                {
                    int groups = *(ushort*)(record + 30);
                    bool usable = affinity is null;
                    for (int g = 0; g < groups && !usable; g++)
                    {
                        byte* groupAffinity = record + 32 + (g * groupSize);
                        ulong mask = maskSize == 8 ? *(ulong*)groupAffinity : *(uint*)groupAffinity;
                        ushort group = *(ushort*)(groupAffinity + maskSize);

                        // affinity は 0 番のグループの分だけ。それ以外のグループのコアは使えるものとみなす
                        usable = group != 0 || (mask & affinity!.Value) != 0;
                    }

                    if (usable)
                    {
                        cores++;
                    }
                }

                offset += size;
            }

            return cores;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern unsafe bool GetLogicalProcessorInformationEx(int relationshipType, byte* buffer, ref uint returnedLength);

    [DllImport("libc", EntryPoint = "sysctlbyname")]
    private static extern int SysctlByName(string name, ref int oldp, ref nint oldlenp, nint newp, nint newlen);
}
