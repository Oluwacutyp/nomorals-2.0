// DevonGhidraExport.java — Devon's headless export script for Ghidra.
//
// Runs inside Ghidra (analyzeHeadless -postScript) and writes ONE
// JSON document describing the analyzed program: identity, the
// function table with call edges, imports, defined strings, and
// bounded decompiler pseudocode for the first functions. Devon's
// Python rung reads that JSON and folds it into Evidence records —
// the engine does the reversing, Devon owns the evidence.
//
// The target is never executed: this is static headless analysis
// over an ephemeral project that the caller deletes afterwards.
// @category DevonRE

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.data.AbstractStringDataType;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.ExternalLocation;
import ghidra.program.model.symbol.ExternalLocationIterator;
import ghidra.program.model.symbol.ExternalManager;
import ghidra.program.util.DefinedDataIterator;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class DevonGhidraExport extends GhidraScript {

    private static final int MAX_FUNCTIONS = 400;
    private static final int MAX_STRINGS = 300;
    private static final int MAX_IMPORTS = 300;
    private static final int MAX_DECOMPILE = 40;
    private static final int MAX_CODE_CHARS = 4000;
    private static final int MAX_CALLEES = 50;
    private static final int DECOMPILE_TIMEOUT_S = 30;

    private static String esc(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder b = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        b.append(String.format("\\u%04x", (int) c));
                    } else {
                        b.append(c);
                    }
            }
        }
        return b.toString();
    }

    private static String q(String s) {
        return "\"" + esc(s) + "\"";
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String outPath = args.length > 0 ? args[0] : "devon_re_export.json";

        StringBuilder j = new StringBuilder(65536);
        j.append("{");
        j.append("\"program\":{");
        j.append("\"name\":").append(q(currentProgram.getName())).append(",");
        j.append("\"executable_format\":")
            .append(q(currentProgram.getExecutableFormat())).append(",");
        j.append("\"language\":")
            .append(q(currentProgram.getLanguage().getLanguageID().toString()))
            .append(",");
        j.append("\"compiler\":")
            .append(q(currentProgram.getCompilerSpec().getCompilerSpecID()
                .toString()))
            .append(",");
        j.append("\"image_base\":")
            .append(q(currentProgram.getImageBase().toString())).append(",");
        j.append("\"min_address\":")
            .append(q(currentProgram.getMinAddress().toString())).append(",");
        j.append("\"max_address\":")
            .append(q(currentProgram.getMaxAddress().toString()));
        j.append("},");

        // ── functions + call edges (+ bounded decompile) ──────────
        DecompInterface ifc = new DecompInterface();
        ifc.setSimplificationStyle("decompile");
        boolean decompilerUp = ifc.openProgram(currentProgram);
        int fnCount = 0;
        int decompiled = 0;
        boolean fnTruncated = false;
        j.append("\"functions\":[");
        FunctionIterator fit =
            currentProgram.getFunctionManager().getFunctions(true);
        boolean firstFn = true;
        while (fit.hasNext()) {
            Function fn = fit.next();
            if (fnCount >= MAX_FUNCTIONS) {
                fnTruncated = true;
                break;
            }
            fnCount++;
            if (!firstFn) {
                j.append(",");
            }
            firstFn = false;
            j.append("{");
            j.append("\"name\":").append(q(fn.getName())).append(",");
            j.append("\"entry\":")
                .append(q(fn.getEntryPoint().toString())).append(",");
            j.append("\"body_size\":")
                .append(fn.getBody().getNumAddresses()).append(",");
            j.append("\"signature\":")
                .append(q(fn.getSignature().toString())).append(",");
            j.append("\"is_thunk\":").append(fn.isThunk()).append(",");
            j.append("\"is_external\":").append(fn.isExternal()).append(",");
            Set<Function> called = fn.getCalledFunctions(monitor);
            j.append("\"calls\":[");
            int ci = 0;
            for (Function callee : called) {
                if (ci >= MAX_CALLEES) {
                    break;
                }
                if (ci > 0) {
                    j.append(",");
                }
                j.append(q(callee.getName()));
                ci++;
            }
            j.append("]");
            if (decompilerUp && !fn.isExternal() && !fn.isThunk()
                    && decompiled < MAX_DECOMPILE) {
                try {
                    DecompileResults res = ifc.decompileFunction(
                        fn, DECOMPILE_TIMEOUT_S, monitor);
                    if (res != null && res.getDecompiledFunction() != null) {
                        String code = res.getDecompiledFunction().getC();
                        if (code != null) {
                            if (code.length() > MAX_CODE_CHARS) {
                                code = code.substring(0, MAX_CODE_CHARS);
                            }
                            j.append(",\"decompile\":").append(q(code));
                            decompiled++;
                        }
                    } else if (res != null) {
                        j.append(",\"decompile_error\":")
                            .append(q(res.getErrorMessage()));
                    }
                } catch (Exception exc) {
                    j.append(",\"decompile_error\":")
                        .append(q(String.valueOf(exc.getMessage())));
                }
            }
            j.append("}");
        }
        j.append("],");
        j.append("\"functions_total\":").append(fnCount).append(",");
        j.append("\"functions_truncated\":").append(fnTruncated).append(",");
        j.append("\"decompiled_count\":").append(decompiled).append(",");
        j.append("\"decompiler_available\":").append(decompilerUp)
            .append(",");

        // ── imports (external locations, grouped by library) ─────
        j.append("\"imports\":[");
        ExternalManager extMgr = currentProgram.getExternalManager();
        int impCount = 0;
        boolean firstImp = true;
        List<String> seenImports = new ArrayList<>();
        for (String lib : extMgr.getExternalLibraryNames()) {
            ExternalLocationIterator lit =
                extMgr.getExternalLocations(lib);
            while (lit.hasNext() && impCount < MAX_IMPORTS) {
                ExternalLocation loc = lit.next();
                String name = loc.getLabel();
                if (name == null && loc.getFunction() != null) {
                    name = loc.getFunction().getName();
                }
                if (name == null) {
                    continue;
                }
                String key = lib + "!" + name;
                if (seenImports.contains(key)) {
                    continue;
                }
                seenImports.add(key);
                impCount++;
                if (!firstImp) {
                    j.append(",");
                }
                firstImp = false;
                j.append("{\"name\":").append(q(name))
                    .append(",\"library\":").append(q(lib)).append("}");
            }
        }
        j.append("],");

        // ── defined strings ───────────────────────────────────────
        j.append("\"strings\":[");
        int strCount = 0;
        boolean firstStr = true;
        boolean strTruncated = false;
        for (Data data : DefinedDataIterator.byDataType(
                currentProgram,
                dt -> dt instanceof AbstractStringDataType)) {
            if (strCount >= MAX_STRINGS) {
                strTruncated = true;
                break;
            }
            StringDataInstance sdi =
                StringDataInstance.getStringDataInstance(data);
            String value = sdi == null ? null : sdi.getStringValue();
            if (value == null) {
                Object raw = data.getValue();
                value = raw == null ? null : raw.toString();
            }
            if (value == null) {
                continue;
            }
            strCount++;
            if (!firstStr) {
                j.append(",");
            }
            firstStr = false;
            j.append("{\"address\":")
                .append(q(data.getAddress().toString()))
                .append(",\"value\":").append(q(value))
                .append("}");
        }
        j.append("],");
        j.append("\"strings_truncated\":").append(strTruncated);
        j.append("}");

        File out = new File(outPath);
        try (FileWriter w = new FileWriter(out)) {
            w.write(j.toString());
        }
        println("DevonGhidraExport: wrote " + outPath + " ("
            + fnCount + " functions, " + decompiled + " decompiled, "
            + impCount + " imports, " + strCount + " strings)");
        ifc.dispose();
    }
}
