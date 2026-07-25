"""Self-contained project checks that preserve the requested naming style."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np


def GetProjectRoot() -> Path:
    """Return the repository root without retaining module-level state.

    Processing details:
        Algorithm: Resolve this test file and select its parent repository
        directory whenever a test needs an absolute project path.

    Returns:
        result: Absolute path containing ``main.py``, ``inc``, and ``doc``.
    """

    return Path(__file__).resolve().parents[1]


if str(GetProjectRoot()) not in sys.path:
    sys.path.insert(0, str(GetProjectRoot()))

from inc.lib.Analysis import Analysis
from inc.lib.DpdIlc import (
    CalculateIterationMetrics,
    FitMimoGmpPredistorter,
    ILCConfig,
    RunFrequencyDomainIlc,
    RunMimoFrequencyDomainIlc,
)
from inc.utils.Draw import Draw
from inc.lib.PaModel import MimoPaModel, PaModel
from inc.utils.ParseWifi import ParseWifi
from inc.utils.SigProc import PowerCalibration, SigProc
from inc.lib.WaveGenWifi import (
    NormalizeFrameFormat,
    WaveGenWifi,
)


def CheckMcsTables() -> None:
    """Verify the complete VHT, HE, and EHT MCS ranges.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    wifiGenerator = WaveGenWifi()
    ehtMcsTable = wifiGenerator.ResolveMcsTable("EHT")
    heMcsTable = wifiGenerator.ResolveMcsTable("HE")
    vhtMcsTable = wifiGenerator.ResolveMcsTable("VHT")
    assert set(ehtMcsTable) == set(range(14))
    assert set(heMcsTable) == set(range(12))
    assert set(vhtMcsTable) == set(range(10))
    assert ehtMcsTable[0].qamOrder == 2
    assert ehtMcsTable[13].qamOrder == 4096
    assert ehtMcsTable[13].codeRate == 5.0 / 6.0
    assert heMcsTable[11].qamOrder == 1024
    assert vhtMcsTable[9].qamOrder == 256


def CheckFrameFormatAliases() -> None:
    """Verify standard names and PHY names resolve to identical formats.

    Processing details:
        Algorithm: Exercise every documented alias in mixed case, generate a
        packet through each public input form, compare its canonical PHY name,
        and require aliases of one generation to produce identical samples.

    Returns:
        result: None. Assertion failures identify alias regressions.
    """

    aliasExpectations = {
        "VHT": "VHT",
        "11ac": "VHT",
        "802.11AC": "VHT",
        "HE": "HE",
        "11ax": "HE",
        "802.11AX": "HE",
        "EHT": "EHT",
        "11be": "EHT",
        "802.11BE": "EHT",
    }
    referenceSamplesByFormat = {}
    for inputName, expectedFormat in aliasExpectations.items():
        assert NormalizeFrameFormat(inputName) == expectedFormat
        maximumMcs = {"VHT": 9, "HE": 11, "EHT": 13}[expectedFormat]
        waveform = WaveGenWifi(
            frameFormat=inputName,
            bandwidthMhz=20,
            mcs=maximumMcs,
            numDataSymbols=1,
            oversampling=1,
        ).Generate()
        assert waveform.frameFormat == expectedFormat
        if expectedFormat not in referenceSamplesByFormat:
            referenceSamplesByFormat[expectedFormat] = waveform.samples
        else:
            assert np.array_equal(
                waveform.samples,
                referenceSamplesByFormat[expectedFormat],
            )


def CheckFunctionStyle() -> None:
    """Verify names, typed signatures, and detailed English documentation.

    Processing details:
        Algorithm: Parse every project Python file with the standard-library
        AST, require ``self`` or ``cls`` for bound methods, require explicit
        parameter and return annotations, allow Python-required
        double-underscore names, and validate multi-line documentation.

    Returns:
        result: None. Assertion failures identify naming or documentation
        regressions before code is published.
    """

    sourceFiles = [GetProjectRoot() / "main.py"]
    sourceFiles.extend(sorted((GetProjectRoot() / "inc").rglob("*.py")))
    sourceFiles.extend(sorted((GetProjectRoot() / "tests").glob("*.py")))
    pascalCasePattern = re.compile(r"[A-Z][A-Za-z0-9]*")
    for sourceFile in sourceFiles:
        syntaxTree = ast.parse(sourceFile.read_text(encoding="utf-8"))
        parentByNode = {}
        for parentNode in ast.walk(syntaxTree):
            for childNode in ast.iter_child_nodes(parentNode):
                parentByNode[childNode] = parentNode
        for syntaxNode in ast.walk(syntaxTree):
            if not isinstance(
                syntaxNode,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            isDoubleUnderscoreMethod = (
                syntaxNode.name.startswith("__")
                and syntaxNode.name.endswith("__")
            )
            assert isDoubleUnderscoreMethod or pascalCasePattern.fullmatch(
                syntaxNode.name
            ), f"function name must be PascalCase: {sourceFile}:{syntaxNode.lineno}"

            positionalArguments = [
                *syntaxNode.args.posonlyargs,
                *syntaxNode.args.args,
            ]
            parentNode = parentByNode.get(syntaxNode)
            if isinstance(parentNode, ast.ClassDef):
                decoratorNames = {
                    decoratorNode.id
                    for decoratorNode in syntaxNode.decorator_list
                    if isinstance(decoratorNode, ast.Name)
                }
                if "staticmethod" not in decoratorNames:
                    expectedFirstArgument = (
                        "cls" if "classmethod" in decoratorNames else "self"
                    )
                    assert positionalArguments, (
                        f"bound method requires {expectedFirstArgument}: "
                        f"{sourceFile}:{syntaxNode.lineno}"
                    )
                    assert (
                        positionalArguments[0].arg == expectedFirstArgument
                    ), (
                        f"bound method first argument must be "
                        f"{expectedFirstArgument}: "
                        f"{sourceFile}:{syntaxNode.lineno}"
                    )

            annotatedArguments = [
                *positionalArguments,
                *syntaxNode.args.kwonlyargs,
            ]
            for argumentNode in annotatedArguments:
                if argumentNode.arg in ("self", "cls"):
                    continue
                assert argumentNode.annotation is not None, (
                    f"missing parameter type annotation for "
                    f"{argumentNode.arg}: "
                    f"{sourceFile}:{syntaxNode.lineno}"
                )
            if syntaxNode.args.vararg is not None:
                assert syntaxNode.args.vararg.annotation is not None, (
                    f"missing variadic parameter type annotation: "
                    f"{sourceFile}:{syntaxNode.lineno}"
                )
            if syntaxNode.args.kwarg is not None:
                assert syntaxNode.args.kwarg.annotation is not None, (
                    f"missing keyword parameter type annotation: "
                    f"{sourceFile}:{syntaxNode.lineno}"
                )
            assert syntaxNode.returns is not None, (
                f"missing return type annotation: "
                f"{sourceFile}:{syntaxNode.lineno}"
            )

            documentation = ast.get_docstring(syntaxNode, clean=False)
            assert documentation is not None, (
                f"missing function documentation: "
                f"{sourceFile}:{syntaxNode.lineno}"
            )
            assert len(documentation.strip().splitlines()) > 1, (
                f"function documentation must be detailed and multi-line: "
                f"{sourceFile}:{syntaxNode.lineno}"
            )


def CheckNoGlobalDataVariables() -> None:
    """Reject module-level data assignments in every project Python file.

    Processing details:
        Algorithm: Parse ``main.py`` plus every ``inc`` and ``tests`` module,
        inspect only the module body, and reject ordinary, annotated,
        augmented, or named assignments while allowing imports, classes,
        functions, and constructor- or function-local configuration data.

    Returns:
        result: None. Assertion failures identify any global data variable
        before it can become shared mutable state or hidden configuration.
    """

    productionFiles = [GetProjectRoot() / "main.py"]
    productionFiles.extend(sorted((GetProjectRoot() / "inc").rglob("*.py")))
    productionFiles.extend(
        sorted((GetProjectRoot() / "tests").glob("*.py"))
    )
    forbiddenAssignmentTypes = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.NamedExpr,
    )
    for sourceFile in productionFiles:
        syntaxTree = ast.parse(sourceFile.read_text(encoding="utf-8"))
        for syntaxNode in syntaxTree.body:
            assert not isinstance(
                syntaxNode, forbiddenAssignmentTypes
            ), (
                f"module-level data variable is forbidden: "
                f"{sourceFile}:{syntaxNode.lineno}"
            )


def CheckModuleResponsibilityBoundaries() -> None:
    """Verify the renamed classes and decoupled module ownership.

    Processing details:
        Algorithm: Inspect production source text and file paths, require each
        moved definition to have one authoritative module, and reject the old
        signal-processing file or Analysis dependencies on generator/PA code.

    Returns:
        result: None. Architecture regressions fail with direct assertions.
    """

    projectRoot = GetProjectRoot()
    analysisSource = (
        projectRoot / "inc" / "lib" / "Analysis.py"
    ).read_text(encoding="utf-8")
    paSource = (projectRoot / "inc" / "lib" / "PaModel.py").read_text(
        encoding="utf-8"
    )
    waveGeneratorSource = (
        projectRoot / "inc" / "lib" / "WaveGenWifi.py"
    ).read_text(encoding="utf-8")
    signalProcessorSource = (
        projectRoot / "inc" / "utils" / "SigProc.py"
    ).read_text(encoding="utf-8")
    frameProcessorSource = (
        projectRoot / "inc" / "utils" / "FrameProcess.py"
    ).read_text(encoding="utf-8")
    metadataSource = (
        projectRoot / "inc" / "utils" / "WifiMetadata.py"
    ).read_text(encoding="utf-8")
    ilcSource = (projectRoot / "inc" / "lib" / "DpdIlc.py").read_text(
        encoding="utf-8"
    )

    assert not (projectRoot / "inc" / "SigProcess.py").exists()
    for movedModuleName in (
        "Analysis.py",
        "DpdIlc.py",
        "PaModel.py",
        "WaveGenWifi.py",
        "Draw.py",
        "FrameProcess.py",
        "ParseWifi.py",
        "SigProc.py",
        "WifiMetadata.py",
        "ConfigUtils.py",
    ):
        assert not (projectRoot / "inc" / movedModuleName).exists()
    assert "class SigProc:" in signalProcessorSource
    assert "class PowerCalibration:" in signalProcessorSource
    assert "class PowerCalibration:" not in paSource
    assert "def BuildCsdPhaseMatrix(" in frameProcessorSource
    assert "def BuildCsdPhaseMatrix(" not in waveGeneratorSource
    assert "class MCSInfo:" in metadataSource
    assert "class WifiWaveform:" in metadataSource
    assert "from ..utils.SigProc import" in analysisSource
    assert "from ..utils.FrameProcess import FrameProcess" in analysisSource
    assert "from ..utils.WifiMetadata import WifiWaveform" in analysisSource
    assert "PaModel import" not in analysisSource
    assert "WaveGenWifi import" not in analysisSource
    assert "evmMseEvaluator" not in ilcSource
    assert "CalculateEvm" not in ilcSource


def CheckBenchmarkSeparation() -> None:
    """Verify that scenario tests are isolated from production ILC code.

    Processing details:
        Algorithm: Inspect both source files, reject benchmark configuration,
        scenario orchestration, or report functions in
        ``inc/lib/DpdIlc.py``, and
        require the standalone test module plus its classified documentation
        to expose every expected scenario and result section.

    Returns:
        result: None. Assertions identify architectural regressions before
        benchmark workflows can leak back into production algorithms.
    """

    ilcSource = (
        GetProjectRoot() / "inc" / "lib" / "DpdIlc.py"
    ).read_text(encoding="utf-8")
    benchmarkPath = GetProjectRoot() / "tests" / "BenchMark.py"
    benchmarkSource = benchmarkPath.read_text(encoding="utf-8")
    benchmarkDocument = (
        GetProjectRoot() / "doc" / "BenchMark.md"
    ).read_text(encoding="utf-8")
    forbiddenProductionNames = (
        "BenchmarkConfig",
        "BenchmarkRow",
        "RunAllIlcBenchmark",
        "SaveBenchmarkResults",
        "PrintBenchmarkResults",
    )
    for forbiddenName in forbiddenProductionNames:
        assert forbiddenName not in ilcSource, (
            f"benchmark workflow leaked into DpdIlc.py: {forbiddenName}"
        )
        assert forbiddenName in benchmarkSource, (
            f"missing benchmark API in tests/BenchMark.py: {forbiddenName}"
        )

    requiredScenarioLabels = (
        "nominal repeated waveform",
        "peak-constrained waveform",
        "32 dB feedback robustness",
        "IQ image impairment",
        "held-out Wi-Fi packet",
    )
    for scenarioLabel in requiredScenarioLabels:
        assert scenarioLabel in benchmarkSource, (
            f"missing benchmark scenario: {scenarioLabel}"
        )

    requiredComparisonMethods = (
        "Peak-constrained baseline",
        "Unconstrained frequency-domain ILC",
        "Naive noisy-feedback ILC",
        "Noise-aware ILC",
        "Frequency-domain ILC on IQ plant",
        "Augmented IQ ILC",
    )
    for methodName in requiredComparisonMethods:
        assert methodName in benchmarkSource, (
            f"missing same-scenario comparison method: {methodName}"
        )

    requiredDocumentSections = (
        "A类：基础对照场景",
        "B类：标称波形更新律场景",
        "C类：约束与噪声鲁棒性场景",
        "D类：IQ失衡增广场景",
        "E类：ILC标签部署泛化场景",
        "F类：功率-EVM扫描场景",
        "BenchMark.py函数级结构与完整执行时序",
        "结果文件字段与审计方法",
        "公平性、可复现性和统计限制",
        "分层验收清单",
        "五种baseline的对比",
        "同场景方法优缺点对比",
        "C类同场景对比结论",
        "D类同场景选择结论",
        "同场景部署模型优缺点对比",
        "功率维度的优缺点对比",
    )
    for sectionTitle in requiredDocumentSections:
        assert sectionTitle in benchmarkDocument, (
            f"missing classified benchmark documentation: {sectionTitle}"
        )


def CheckFunctionPrincipleCoverage() -> None:
    """Verify that every production function has an exact audit entry.

    Processing details:
        Algorithm: Build parent links for each production AST, qualify every
        function by its owning class or module, and require the resulting name
        to appear verbatim in the function-principle audit document.

    Returns:
        result: None. Missing documentation mappings fail the project checks.
    """

    auditPath = GetProjectRoot() / "doc" / "FunctionPrinciples.md"
    auditText = auditPath.read_text(encoding="utf-8")
    sourceFiles = [GetProjectRoot() / "main.py"]
    sourceFiles.extend(sorted((GetProjectRoot() / "inc").rglob("*.py")))
    checkedDefinitionCount = 0
    for sourceFile in sourceFiles:
        syntaxTree = ast.parse(sourceFile.read_text(encoding="utf-8"))
        parentByNode = {}
        for parentNode in ast.walk(syntaxTree):
            for childNode in ast.iter_child_nodes(parentNode):
                parentByNode[childNode] = parentNode
        for syntaxNode in ast.walk(syntaxTree):
            if not isinstance(
                syntaxNode,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            ownerName = sourceFile.stem
            ancestorNode = parentByNode.get(syntaxNode)
            while ancestorNode is not None:
                if isinstance(ancestorNode, ast.ClassDef):
                    ownerName = ancestorNode.name
                    break
                ancestorNode = parentByNode.get(ancestorNode)
            qualifiedName = f"{ownerName}.{syntaxNode.name}"
            assert f"`{qualifiedName}`" in auditText, (
                f"missing function-principle mapping: {qualifiedName} "
                f"at {sourceFile}:{syntaxNode.lineno}"
            )
            checkedDefinitionCount += 1
    assert checkedDefinitionCount >= 186


def CheckDocumentationMathCompatibility() -> None:
    """Verify that every principle document uses portable math syntax.

    Processing details:
        Algorithm: Scan every Markdown document for unsupported macros,
        invisible control characters, legacy display delimiters, incomplete
        math fences, broken inline delimiters, fragile inline ellipses, and
        unbalanced braces inside both inline and fenced equations.

    Returns:
        result: None. Assertion failures identify the affected document and
        equation before incompatible formulas can be published.
    """

    documentPaths = sorted((GetProjectRoot() / "doc").glob("*.md"))
    forbiddenMacros = (r"\operatorname", r"\text", r"\dfrac")
    fenceMarker = chr(96) * 3
    mathFenceMarker = fenceMarker + "math"
    mathBlockPattern = (
        re.escape(mathFenceMarker)
        + r"[ \t]*\r?\n(.*?)"
        + re.escape(fenceMarker)
    )
    assert documentPaths
    for documentPath in documentPaths:
        markdownText = documentPath.read_text(encoding="utf-8")
        controlCharacters = [
            character
            for character in markdownText
            if ord(character) < 32 and character not in "\n\r\t"
        ]
        assert not controlCharacters, (
            f"control character in documentation: {documentPath}"
        )
        assert "$$" not in markdownText, (
            f"legacy display-math delimiter in documentation: {documentPath}"
        )
        for forbiddenMacro in forbiddenMacros:
            assert forbiddenMacro not in markdownText, (
                f"unsupported math macro {forbiddenMacro}: {documentPath}"
            )

        insideCodeFence = False
        for lineNumber, markdownLine in enumerate(
            markdownText.splitlines(),
            start=1,
        ):
            if markdownLine.startswith(fenceMarker):
                insideCodeFence = not insideCodeFence
                continue
            if insideCodeFence:
                continue

            inlineDelimiterCount = len(
                re.findall(r"(?<!\\)\$", markdownLine)
            )
            assert inlineDelimiterCount % 2 == 0, (
                f"broken inline math delimiter in {documentPath}:"
                f"{lineNumber}"
            )
            inlineMathFragments = re.findall(
                r"(?<!\\)\$([^$\r\n]+)(?<!\\)\$",
                markdownLine,
            )
            for inlineMath in inlineMathFragments:
                assert r"\ldots" not in inlineMath, (
                    f"inline sequence must use a math block in "
                    f"{documentPath}:{lineNumber}"
                )
                braceDepth = 0
                for characterIndex, character in enumerate(inlineMath):
                    isEscaped = (
                        characterIndex > 0
                        and inlineMath[characterIndex - 1] == "\\"
                    )
                    if character == "{" and not isEscaped:
                        braceDepth += 1
                    elif character == "}" and not isEscaped:
                        braceDepth -= 1
                    assert braceDepth >= 0, (
                        f"unexpected inline closing brace in "
                        f"{documentPath}:{lineNumber}"
                    )
                assert braceDepth == 0, (
                    f"unbalanced inline braces in "
                    f"{documentPath}:{lineNumber}"
                )
            lineWithoutInlineMath = re.sub(
                r"(?<!\\)\$[^$\r\n]+(?<!\\)\$",
                "",
                markdownLine,
            )
            assert re.search(
                r"\\[A-Za-z]+",
                lineWithoutInlineMath,
            ) is None, (
                f"math macro outside a delimiter in "
                f"{documentPath}:{lineNumber}"
            )

        mathBlocks = re.findall(
            mathBlockPattern,
            markdownText,
            flags=re.DOTALL,
        )
        assert len(mathBlocks) == markdownText.count(mathFenceMarker), (
            f"incomplete math fence in documentation: {documentPath}"
        )
        for blockIndex, mathBlock in enumerate(mathBlocks, start=1):
            braceDepth = 0
            for characterIndex, character in enumerate(mathBlock):
                isEscaped = (
                    characterIndex > 0
                    and mathBlock[characterIndex - 1] == "\\"
                )
                if character == "{" and not isEscaped:
                    braceDepth += 1
                elif character == "}" and not isEscaped:
                    braceDepth -= 1
                assert braceDepth >= 0, (
                    f"unexpected closing brace in {documentPath}, "
                    f"math block {blockIndex}"
                )
            assert braceDepth == 0, (
                f"unbalanced braces in {documentPath}, math block {blockIndex}"
            )


def CheckInternalDefaultConfiguration() -> None:
    """Verify internal defaults, live external edits, and direct overrides.

    Processing details:
        Algorithm: Pass only caller-owned override dictionaries to each class,
        verify that omitted values come from constructor-internal ChainMap
        defaults, and ensure external edits remain visible.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    externalWifiParameters = {
        "bandwidthMhz": 20,
        "mcs": 0,
        "numDataSymbols": 1,
        "oversampling": 4,
    }
    wifiGenerator = WaveGenWifi(parameters=externalWifiParameters)
    assert wifiGenerator.frameFormat == "EHT"
    assert wifiGenerator.seed == 7

    externalWifiParameters["frameFormat"] = "HE"
    externalWifiParameters["seed"] = 29
    assert wifiGenerator.Generate().frameFormat == "HE"
    assert wifiGenerator.seed == 29

    wifiGenerator.UpdateParameters(bandwidthMhz=40)
    externalWifiParameters["bandwidthMhz"] = 80
    assert wifiGenerator.bandwidthMhz == 40
    try:
        wifiGenerator.UpdateParameters(mcs=99)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid parameter overrides must be rejected")
    assert wifiGenerator.mcs == 0

    externalPaParameters = {"modelName": "wiener"}
    paModel = PaModel(parameters=externalPaParameters)
    assert paModel.modelName == "wiener"
    externalPaParameters["modelName"] = "gmp"
    paModel.Process(np.array([0.1 + 0.0j], dtype=np.complex128))
    assert paModel.modelName == "gmp"
    assert paModel.model.__class__.__name__ == "GMPPA"

    analysisParameters = {"maxSegmentLength": 1024}
    analysisWaveform = WaveGenWifi(
        parameters={
            "bandwidthMhz": 20,
            "mcs": 0,
            "numDataSymbols": 2,
            "oversampling": 4,
        }
    ).Generate()
    resultAnalysis = Analysis(
        analysisWaveform.samples,
        analysisWaveform,
        parameters=analysisParameters,
    )
    assert resultAnalysis.GetParameters()["powerEvmFileStem"] == (
        "power_evm_curve"
    )
    analysisParameters["powerEvmFileStem"] = "external_curve"
    assert resultAnalysis.GetParameters()["powerEvmFileStem"] == (
        "external_curve"
    )
    resultAnalysis.CalculateAclr(analysisWaveform.samples)

    externalDrawParameters = {"figureDpi": 100}
    resultDraw = Draw(parameters=externalDrawParameters)
    assert resultDraw.GetParameters()["powerEvmFileStem"] == (
        "power_evm_curve"
    )
    externalDrawParameters["powerEvmFileStem"] = "external_figure"
    assert resultDraw.GetParameters()["powerEvmFileStem"] == (
        "external_figure"
    )

    # Production call sites must not reconstruct internal default layers.
    for relativePath in ("main.py", "inc/lib/DpdIlc.py"):
        callSiteSource = (GetProjectRoot() / relativePath).read_text(
            encoding="utf-8"
        )
        assert "ChainMap" not in callSiteSource
        assert "DefaultParameters" not in callSiteSource


def CheckWifiFormats() -> None:
    """Verify that each generator instance creates its selected frame format.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    formatExpectations = {
        "VHT": (
            "L-STF",
            "L-LTF",
            "L-SIG",
            "VHT-SIG-A",
            "VHT-STF",
            "VHT-LTF",
            "VHT-SIG-B",
            "VHT-Data",
        ),
        "EHT": (
            "L-STF",
            "L-LTF",
            "L-SIG",
            "RL-SIG",
            "U-SIG",
            "EHT-SIG",
            "EHT-STF",
            "EHT-LTF",
            "EHT-Data",
        ),
        "HE": (
            "L-STF",
            "L-LTF",
            "L-SIG",
            "RL-SIG",
            "HE-SIG-A",
            "HE-STF",
            "HE-LTF",
            "HE-Data",
        ),
    }
    for frameFormat, expectedFields in formatExpectations.items():
        wifiGenerator = WaveGenWifi(
            frameFormat=frameFormat,
            bandwidthMhz=20,
            mcs=0,
            numDataSymbols=2,
            oversampling=1,
        )
        waveform = wifiGenerator.Generate()
        assert waveform.frameFormat == frameFormat
        assert waveform.dataFieldName == f"{frameFormat}-Data"
        assert tuple(waveform.fieldSlices) == expectedFields

        # Verify common fixed field durations at the configured sample rate.
        assert (
            waveform.fieldSlices["L-STF"].stop
            - waveform.fieldSlices["L-STF"].start
        ) == int(round(8e-6 * waveform.sampleRateHz))
        assert (
            waveform.fieldSlices[f"{frameFormat}-STF"].stop
            - waveform.fieldSlices[f"{frameFormat}-STF"].start
        ) == int(round(4e-6 * waveform.sampleRateHz))


def CheckWifiBandwidths() -> None:
    """Verify VHT and HE/EHT FFT, data-tone, and pilot-tone counts.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    expectedValuesByFormat = {
        "VHT": {
            20: (64, 52, 4),
            40: (128, 108, 6),
            80: (256, 234, 8),
            160: (512, 468, 16),
        },
        "HE": {
            20: (256, 234, 8),
            40: (512, 468, 16),
            80: (1024, 980, 16),
            160: (2048, 1960, 32),
        },
        "EHT": {
            20: (256, 234, 8),
            40: (512, 468, 16),
            80: (1024, 980, 16),
            160: (2048, 1960, 32),
        },
    }
    for frameFormat, expectedValues in expectedValuesByFormat.items():
        for bandwidthMhz, (
            baseFftLength,
            dataToneCount,
            pilotToneCount,
        ) in expectedValues.items():
            wifiGenerator = WaveGenWifi(
                frameFormat=frameFormat,
                bandwidthMhz=bandwidthMhz,
                mcs=0,
                numDataSymbols=2,
                oversampling=1,
            )
            waveform = wifiGenerator.Generate()
            assert waveform.fftLength == baseFftLength
            assert waveform.dataSubcarriers.size == dataToneCount
            assert waveform.pilotSubcarriers.size == pilotToneCount


def CheckSampleRateConfiguration() -> None:
    """Verify direct sample-rate control and legacy oversampling fallback.

    Processing details:
        Algorithm: Generate VHT and EHT packets at compatible noninteger
        bandwidth ratios, require the requested clock to determine FFT and
        guard lengths, confirm ``sampleRateHz`` overrides the legacy factor,
        and reject clocks that cannot represent exact OFDM timing intervals.

    Returns:
        result: None. Assertions identify sample-clock configuration
        regressions before waveform metadata reaches analysis or ILC.
    """

    ehtWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=2,
        sampleRateHz=50.0e6,
        oversampling=8,
    ).Generate()
    assert ehtWaveform.sampleRateHz == 50.0e6
    assert ehtWaveform.oversampling == 2.5
    assert ehtWaveform.fftLength == 640
    assert ehtWaveform.cpLength == 40

    vhtWaveform = WaveGenWifi(
        frameFormat="VHT",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=2,
        guardIntervalUs=0.4,
        sampleRateHz=30.0e6,
    ).Generate()
    assert vhtWaveform.sampleRateHz == 30.0e6
    assert vhtWaveform.oversampling == 1.5
    assert vhtWaveform.fftLength == 96
    assert vhtWaveform.cpLength == 12

    legacyGenerator = WaveGenWifi(
        bandwidthMhz=20,
        mcs=0,
        numDataSymbols=1,
        oversampling=3,
    )
    assert legacyGenerator.sampleRateHz == 60.0e6
    assert legacyGenerator.GetParameters()["sampleRateHz"] == 60.0e6

    try:
        WaveGenWifi(
            frameFormat="EHT",
            bandwidthMhz=20,
            sampleRateHz=61.44e6,
        )
    except ValueError as error:
        assert "integer sample count" in str(error)
    else:
        raise AssertionError(
            "incompatible sampleRateHz must be rejected"
        )


def CheckMimoSpatialStructure() -> None:
    """Verify VHT/HE/EHT streams, mapping, CSD, and LTF dimensions.

    Processing details:
        Algorithm: Generate representative multi-stream packets for every
        PHY, require an orthonormal antenna mapping, confirm independent data
        dimensions, and invert mapping/CSD through the analysis demodulator.

    Returns:
        result: None. Assertions identify MIMO structure regressions.
    """

    formatCases = (
        ("VHT", 2, 2, 2),
        ("HE", 4, 3, 4),
        ("EHT", 4, 4, 4),
    )
    for frameFormat, transmitCount, streamCount, ltfCount in formatCases:
        waveform = WaveGenWifi(
            frameFormat=frameFormat,
            bandwidthMhz=20,
            mcs=3,
            numDataSymbols=2,
            oversampling=4,
            numTransmitAntennas=transmitCount,
            numSpatialStreams=streamCount,
            spatialMapping="dft",
            seed=101,
        ).Generate()
        assert waveform.samples.shape[1] == transmitCount
        assert waveform.referenceDataSymbols.shape[2] == streamCount
        assert waveform.ltfSymbolCount == ltfCount
        assert waveform.cyclicShiftsSeconds.size == transmitCount
        assert np.allclose(
            waveform.spatialMappingMatrix.conj().T
            @ waveform.spatialMappingMatrix,
            np.eye(streamCount),
            atol=1e-12,
        )
        resultAnalysis = Analysis(waveform.samples, waveform)
        recoveredSymbols = resultAnalysis.DemodulatePreparedWifiData(
            waveform.samples
        )
        assert np.allclose(
            recoveredSymbols,
            waveform.referenceDataSymbols,
            atol=1e-11,
        )
        idealMetrics = resultAnalysis.Analyze(waveform.samples)
        mimoMetrics = resultAnalysis.GetLastMimoMetrics()
        assert isinstance(idealMetrics, dict)
        assert idealMetrics["evmDb"] < -250.0
        assert mimoMetrics is not None
        assert isinstance(mimoMetrics, dict)
        assert (
            len(mimoMetrics["evmDbPerSpatialStream"])
            == streamCount
        )

    # Standard-generation stream limits are enforced independently from the
    # number of physical antennas available to the caller.
    for frameFormat in ("VHT", "HE", "EHT"):
        WaveGenWifi(
            frameFormat=frameFormat,
            numTransmitAntennas=8,
            numSpatialStreams=8,
        )
        try:
            WaveGenWifi(
                frameFormat=frameFormat,
                numTransmitAntennas=9,
                numSpatialStreams=9,
            )
        except ValueError as error:
            assert "1 through 8" in str(error)
        else:
            raise AssertionError(
                f"{frameFormat} must reject more than eight streams"
            )


def CheckMimoPaAndDpd() -> None:
    """Verify independent PA power control and matrix DPD processing.

    Processing details:
        Algorithm: Drive equal chain inputs through equal PA models, verify
        relative dB and absolute dBm controls, then exercise one short ILC and
        fitted-GMP pass while preserving samples-by-chains shapes.

    Returns:
        result: None. Assertions cover power calibration and MIMO DPD APIs.
    """

    sampleIndices = np.arange(2048, dtype=float)
    testVector = 0.08 * np.exp(1j * 2.0 * np.pi * sampleIndices / 37.0)
    testMatrix = np.column_stack((testVector, testVector))
    mimoPaModel = MimoPaModel(
        numTransmitChains=2,
        outputPowerDbPerChain=(0.0, -6.0),
    )
    relativeOutput = mimoPaModel.Process(testMatrix)
    relativeRms = mimoPaModel.GetOutputRmsPerChain()
    assert relativeOutput.shape == testMatrix.shape
    assert np.isclose(
        relativeRms[1] / relativeRms[0],
        10.0 ** (-6.0 / 20.0),
        rtol=1e-12,
    )
    powerCalibration = PowerCalibration(loadResistanceOhm=50.0)
    targetOutputPowerDbm = (
        powerCalibration.RmsToDbm(0.12),
        powerCalibration.RmsToDbm(0.21),
    )
    mimoPaModel.SetTargetOutputPowerDbm(0, targetOutputPowerDbm[0])
    mimoPaModel.SetTargetOutputPowerDbm(1, targetOutputPowerDbm[1])
    mimoPaModel.Process(testMatrix)
    assert np.allclose(
        mimoPaModel.GetOutputRmsPerChain(), (0.12, 0.21), atol=1e-12
    )
    assert np.allclose(
        mimoPaModel.GetOutputPowerDbmPerChain(),
        targetOutputPowerDbm,
        atol=1e-12,
    )
    try:
        mimoPaModel.SetTargetOutputPowerDbm(0, 25.1)
    except ValueError as error:
        assert "maximumOutputPowerDbm" in str(error)
    else:
        raise AssertionError(
            "MIMO output power above the 25 dBm limit must fail"
        )

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=2,
        numDataSymbols=2,
        oversampling=4,
        numTransmitAntennas=2,
        numSpatialStreams=2,
    ).Generate()
    referenceSignal = 0.18 * waveform.samples
    # Disable absolute normalization for a meaningful repeatable ILC plant.
    mimoPaModel.UpdateParameters(
        targetOutputPowerDbmPerChain=(None, None)
    )
    ilcResult = RunMimoFrequencyDomainIlc(
        referenceSignal,
        mimoPaModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(numIterations=2),
    )
    assert ilcResult.learnedInput.shape == referenceSignal.shape
    assert ilcResult.outputSignal.shape == referenceSignal.shape
    assert len(ilcResult.chainResults) == 2
    resultAnalysis = Analysis(referenceSignal, waveform)
    mimoAnalysisResult = resultAnalysis.AnalyzeMimoIlcHistory(
        tuple(
            chainResult.history
            for chainResult in ilcResult.chainResults
        )
    )
    assert len(mimoAnalysisResult.history) == 2
    assert mimoAnalysisResult.bestInputSignal.shape == referenceSignal.shape
    assert mimoAnalysisResult.bestOutputSignal.shape == referenceSignal.shape
    predistorter = FitMimoGmpPredistorter(
        referenceSignal, mimoAnalysisResult.bestInputSignal
    )
    assert predistorter.Process(referenceSignal).shape == referenceSignal.shape


def CheckFormatSpecificMcsValidation() -> None:
    """Verify each PHY rejects MCS and GI values introduced by later PHYs.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    try:
        WaveGenWifi(frameFormat="HE", mcs=12).Generate()
    except ValueError as error:
        assert "HE MCS" in str(error)
    else:
        raise AssertionError("HE MCS 12 must be rejected")

    try:
        WaveGenWifi(frameFormat="11ac", mcs=10).Generate()
    except ValueError as error:
        assert "VHT MCS" in str(error)
    else:
        raise AssertionError("VHT MCS 10 must be rejected")

    try:
        WaveGenWifi(
            frameFormat="VHT", mcs=9, guardIntervalUs=1.6
        ).Generate()
    except ValueError as error:
        assert "VHT guardIntervalUs" in str(error)
    else:
        raise AssertionError("VHT GI 1.6 us must be rejected")


def CheckIdealMetrics() -> None:
    """Verify that a perfect signal path has effectively zero EVM.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    for frameFormat, mcs in (("EHT", 13), ("HE", 11), ("VHT", 9)):
        wifiGenerator = WaveGenWifi(
            frameFormat=frameFormat,
            mcs=mcs,
            numDataSymbols=4,
            oversampling=4,
        )
        waveform = wifiGenerator.Generate()
        resultAnalysis = Analysis(waveform.samples, waveform)
        metrics = resultAnalysis.Analyze(waveform.samples)
        assert isinstance(metrics, dict)
        assert set(metrics) == {
            "snrDb",
            "evmDb",
            "evmPercent",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
        }
        assert not hasattr(metrics, "ToDict")
        assert metrics["snrDb"] > 250.0
        assert metrics["evmDb"] < -250.0
        assert metrics["evmPercent"] < 1e-10


def CheckSignalProcessingCompensation() -> None:
    """Verify joint timing, frequency, and complex-gain compensation.

    Processing details:
        Algorithm: Synthesize a measurement with known integer and fractional
        delay, carrier offset, sample-rate offset, and complex gain; require
        ``SigProc`` to recover each value and ``Analysis`` to consume the
        same utility path before calculating metrics.

    Returns:
        result: None. Assertions bound estimator error and residual EVM.
    """

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=9,
        numDataSymbols=6,
        oversampling=4,
        seed=77,
    ).Generate()
    referenceSignal = 0.20 * waveform.samples
    signalProcessingParameters = {
        "maxIntegerDelaySamples": 64,
        "maxSamplingFrequencyOffsetPpm": 100.0,
        "timingWindowLength": 1024,
        "interpolationHalfLength": 16,
    }
    signalProcessor = SigProc(
        referenceSignal,
        waveform.sampleRateHz,
        parameters=signalProcessingParameters,
    )
    expectedIntegerDelay = 7
    expectedFractionalDelay = 0.28
    expectedCarrierOffsetHz = 25000.0
    expectedSamplingOffsetPpm = 40.0
    expectedComplexGain = 0.73 * np.exp(1j * 0.42)
    measuredIndices = np.arange(referenceSignal.size + 32, dtype=float)
    referencePositions = (
        measuredIndices
        - expectedIntegerDelay
        - expectedFractionalDelay
    ) / (1.0 + expectedSamplingOffsetPpm / 1.0e6)
    measuredSignal = signalProcessor.InterpolateSignal(
        referenceSignal, referencePositions
    )
    measuredSignal *= expectedComplexGain * np.exp(
        1j
        * 2.0
        * np.pi
        * expectedCarrierOffsetHz
        * measuredIndices
        / waveform.sampleRateHz
    )

    processingResult = signalProcessor.Process(
        measuredSignal,
        estimationSlice=waveform.fieldSlices[waveform.dataFieldName],
    )
    assert processingResult.integerDelaySamples == expectedIntegerDelay
    assert abs(
        processingResult.fractionalDelaySamples
        - expectedFractionalDelay
    ) < 0.08
    assert abs(
        processingResult.carrierFrequencyOffsetHz
        - expectedCarrierOffsetHz
    ) < 500.0
    assert abs(
        processingResult.samplingFrequencyOffsetPpm
        - expectedSamplingOffsetPpm
    ) < 5.0
    assert abs(abs(processingResult.complexGain) - abs(expectedComplexGain)) < 0.03
    residualRatio = np.sqrt(
        np.sum(
            np.abs(processingResult.processedSignal - referenceSignal) ** 2
        )
        / np.sum(np.abs(referenceSignal) ** 2)
    )
    assert residualRatio < 0.03

    resultAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "signalProcessingParameters": signalProcessingParameters
        },
    )
    metrics = resultAnalysis.Analyze(measuredSignal)
    assert metrics["evmDb"] < -30.0
    assert resultAnalysis.GetLastSignalProcessingResult() is not None
    resultAnalysis.AnalyzeStages({"Impaired": measuredSignal})
    assert "Impaired" in resultAnalysis.GetStageSignalProcessingResults()
    with TemporaryDirectory() as temporaryDirectory:
        jsonPath, csvPath = resultAnalysis.Save(
            Path(temporaryDirectory),
            {"test": "signal processing"},
        )
        savedPayload = json.loads(jsonPath.read_text(encoding="utf-8"))
        assert "Impaired" in savedPayload["signalProcessing"]
        assert "carrierFrequencyOffsetHz" in csvPath.read_text(
            encoding="utf-8-sig"
        )
    analysisSource = (
        GetProjectRoot() / "inc" / "lib" / "Analysis.py"
    ).read_text(
        encoding="utf-8"
    )
    assert "from ..utils.SigProc import" in analysisSource
    assert "PaModel import" not in analysisSource
    assert "WaveGenWifi import" not in analysisSource
    assert "def BestComplexGain" not in analysisSource


def CheckPowerEvmCurve() -> None:
    """Verify multi-method power-EVM analysis and all saved file formats.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    wifiGenerator = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=2,
        oversampling=4,
        seed=43,
    )
    waveform = wifiGenerator.Generate()
    powerCalibration = PowerCalibration(loadResistanceOhm=50.0)
    assert np.isclose(
        powerCalibration.DbmToRms(0.0),
        np.sqrt(0.001 * 50.0),
    )
    assert np.isclose(
        powerCalibration.RmsToDbm(np.sqrt(0.001 * 50.0)),
        0.0,
    )
    assert powerCalibration.maximumOutputPowerDbm == 25.0
    assert np.isclose(
        powerCalibration.OutputPowerToDriveScale(20.0),
        10.0 ** (-5.0 / 20.0),
    )
    nominalReference = (
        powerCalibration.OutputPowerToDriveScale(20.0)
        * waveform.samples
    )
    paModel = PaModel(modelName="wiener")
    resultAnalysis = Analysis(
        nominalReference,
        waveform,
        loadResistanceOhm=powerCalibration.loadResistanceOhm,
    )
    outputPowerDbmValues = (10.0, 20.0, 25.0)
    curve = resultAnalysis.AnalyzePowerEvmCurve(
        outputPowerDbmValues,
        {
            "Ideal": lambda pointReference, _: pointReference,
            "PA baseline": lambda pointReference, _: paModel.Process(
                pointReference
            ),
        },
    )
    assert curve.outputPowerDbmValues.size == 3
    assert np.allclose(
        curve.outputPowerDbmValues,
        outputPowerDbmValues,
    )
    assert np.allclose(
        curve.driveScaleValues,
        tuple(
            10.0 ** ((powerDbm - 25.0) / 20.0)
            for powerDbm in outputPowerDbmValues
        ),
    )
    assert np.allclose(
        curve.targetOutputRmsValues,
        tuple(
            powerCalibration.DbmToRms(powerDbm)
            for powerDbm in outputPowerDbmValues
        ),
    )
    assert set(curve.evmDbByMethod) == {"Ideal", "PA baseline"}
    assert np.all(curve.evmDbByMethod["Ideal"] < -250.0)
    calibratedOutput = powerCalibration.ScaleSignalToOutputPower(
        paModel.Process(nominalReference),
        20.0,
    )
    calibratedRms = float(
        np.sqrt(np.mean(np.abs(calibratedOutput) ** 2))
    )
    assert np.isclose(
        powerCalibration.RmsToDbm(calibratedRms),
        20.0,
    )
    try:
        powerCalibration.OutputPowerToDriveScale(25.1)
    except ValueError as error:
        assert "maximumOutputPowerDbm" in str(error)
    else:
        raise AssertionError("output power above 25 dBm must fail")

    with TemporaryDirectory() as temporaryDirectory:
        dataPaths = resultAnalysis.SavePowerEvmCurveData(
            Path(temporaryDirectory)
        )
        figurePath = Draw().SavePowerEvmCurve(
            curve,
            Path(temporaryDirectory),
        )
        assert all(outputPath.is_file() for outputPath in dataPaths)
        assert figurePath.is_file()

    analysisSource = (
        GetProjectRoot() / "inc" / "lib" / "Analysis.py"
    ).read_text(
        encoding="utf-8"
    )
    assert "matplotlib" not in analysisSource
    assert ".plot(" not in analysisSource


def CheckGuardIntervals() -> None:
    """Verify compatible 2x/4x long-training durations for every GI.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    expectedLtfDurationUs = {0.8: 13.6, 1.6: 8.0, 3.2: 16.0}
    for frameFormat in ("EHT", "HE"):
        for guardIntervalUs, ltfDurationUs in expectedLtfDurationUs.items():
            wifiGenerator = WaveGenWifi(
                frameFormat=frameFormat,
                bandwidthMhz=20,
                mcs=0,
                numDataSymbols=1,
                guardIntervalUs=guardIntervalUs,
                oversampling=1,
            )
            waveform = wifiGenerator.Generate()
            ltfSlice = waveform.fieldSlices[f"{frameFormat}-LTF"]
            ltfSampleCount = ltfSlice.stop - ltfSlice.start
            assert ltfSampleCount == int(
                round(ltfDurationUs * 1e-6 * waveform.sampleRateHz)
            )

    for guardIntervalUs in (0.4, 0.8):
        waveform = WaveGenWifi(
            frameFormat="11ac",
            bandwidthMhz=20,
            mcs=9,
            numDataSymbols=1,
            guardIntervalUs=guardIntervalUs,
            oversampling=1,
        ).Generate()
        ltfSlice = waveform.fieldSlices["VHT-LTF"]
        assert ltfSlice.stop - ltfSlice.start == int(
            round(4.0e-6 * waveform.sampleRateHz)
        )
        expectedSymbolDurationUs = 3.2 + guardIntervalUs
        assert waveform.symbolLength == int(
            round(expectedSymbolDurationUs * 1e-6 * waveform.sampleRateHz)
        )


def CheckIlcImprovement() -> None:
    """Verify that ILC reduces reconstruction error for both PA families.

    Processing details:
        Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    wifiGenerator = WaveGenWifi(
        frameFormat="HE",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=6,
        oversampling=4,
        seed=31,
    )
    waveform = wifiGenerator.Generate()
    referenceSignal = 0.28 * waveform.samples
    resultAnalysis = Analysis(referenceSignal, waveform)
    for modelName in ("wiener", "gmp"):
        paModel = PaModel(modelName=modelName)
        baselineOutput = paModel.Process(referenceSignal)
        baselineMetrics = resultAnalysis.Analyze(baselineOutput)
        ilcResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            ILCConfig(
                numIterations=6,
                learningRate=0.35,
                maxAmplitude=1.25,
            ),
        )
        ilcAnalysisResult = resultAnalysis.AnalyzeIlcHistory(
            ilcResult.history
        )
        ilcMetrics = ilcAnalysisResult.bestMetrics
        assert ilcMetrics["evmDb"] < baselineMetrics["evmDb"]
        assert ilcMetrics["snrDb"] > baselineMetrics["snrDb"]


def CheckReceiveOnlyWifiAnalysis() -> None:
    """Verify frame parsing and zero-reference Analysis operation.

    Processing details:
        Algorithm: Generate each PHY family with its maximum MCS, pass a
        nonlinear PA output through both the original reference-aided path and
        the new receive-only path, compare every metric, verify packet-offset
        and sample-rate recovery, then repeat the parser check for 2x2 MIMO.

    Returns:
        result: None. Assertions enforce exact metadata recovery and equivalent
            performance calculations between both public Analysis workflows.
    """

    maximumMcsByFormat = {
        "VHT": 9,
        "HE": 11,
        "EHT": 13,
    }
    for formatIndex, (frameFormat, maximumMcs) in enumerate(
        maximumMcsByFormat.items()
    ):
        waveform = WaveGenWifi(
            frameFormat=frameFormat,
            bandwidthMhz=20,
            mcs=maximumMcs,
            numDataSymbols=2,
            sampleRateHz=80.0e6,
            seed=501 + formatIndex,
        ).Generate()
        assert waveform.seed == 501 + formatIndex
        assert waveform.cyclicShiftEnabled
        measuredSignal = PaModel(modelName="wiener").Process(
            0.22 * waveform.samples
        )
        referenceMetrics = Analysis(
            waveform.samples, waveform
        ).Analyze(measuredSignal)
        leadingSamples = 13 + formatIndex
        receiveCapture = np.r_[
            np.zeros(leadingSamples, dtype=np.complex128),
            measuredSignal,
            np.zeros(7, dtype=np.complex128),
        ]
        receiveAnalysis = Analysis(
            receiveCapture,
            parseParameters={"sampleRateHz": 80.0e6},
        )
        receiveMetrics = receiveAnalysis.Analyze()
        parsedFrame = receiveAnalysis.GetParsedWifiFrame()
        assert parsedFrame is not None
        assert parsedFrame.packetStartSample == leadingSamples
        assert parsedFrame.detectedParameters["frameFormat"] == frameFormat
        assert parsedFrame.detectedParameters["mcs"] == maximumMcs
        assert parsedFrame.detectedParameters["numDataSymbols"] == 2
        assert parsedFrame.parseConfidence > 0.95
        assert np.allclose(
            np.asarray(tuple(referenceMetrics.values())),
            np.asarray(tuple(receiveMetrics.values())),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        assistedAnalysis = Analysis(
            receiveCapture,
            transmittedSignal=waveform.samples,
        )
        assistedMetrics = assistedAnalysis.Analyze()
        assistedFrame = assistedAnalysis.GetParsedWifiFrame()
        assert assistedFrame is not None
        assert assistedFrame.packetStartSample == leadingSamples
        assert assistedFrame.parseConfidence > 0.90
        assert np.allclose(
            np.asarray(tuple(referenceMetrics.values())),
            np.asarray(tuple(assistedMetrics.values())),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        if formatIndex == 0:
            objectAssistedAnalysis = Analysis(
                receiveCapture,
                transmittedSignal=waveform,
            )
            objectAssistedMetrics = objectAssistedAnalysis.Analyze()
            objectAssistedFrame = (
                objectAssistedAnalysis.GetParsedWifiFrame()
            )
            assert objectAssistedFrame is not None
            assert objectAssistedFrame.packetStartSample == leadingSamples
            assert np.allclose(
                np.asarray(tuple(referenceMetrics.values())),
                np.asarray(tuple(objectAssistedMetrics.values())),
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            receivedWaveform = replace(
                waveform,
                samples=receiveCapture,
            )
            objectReceiveArrayTransmitAnalysis = Analysis(
                receivedWaveform,
                transmittedSignal=waveform.samples,
            )
            objectReceiveArrayTransmitMetrics = (
                objectReceiveArrayTransmitAnalysis.Analyze()
            )
            objectReceiveObjectTransmitAnalysis = Analysis(
                receivedWaveform,
                transmittedSignal=waveform,
            )
            objectReceiveObjectTransmitMetrics = (
                objectReceiveObjectTransmitAnalysis.Analyze()
            )
            for typeDispatchedMetrics in (
                objectReceiveArrayTransmitMetrics,
                objectReceiveObjectTransmitMetrics,
            ):
                assert np.allclose(
                    np.asarray(tuple(referenceMetrics.values())),
                    np.asarray(tuple(typeDispatchedMetrics.values())),
                    rtol=1.0e-10,
                    atol=1.0e-10,
                )

    autoWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=40,
        mcs=5,
        numDataSymbols=1,
        sampleRateHz=160.0e6,
        seed=514,
    ).Generate()
    autoParsedFrame = ParseWifi().Parse(autoWaveform.samples)
    assert autoParsedFrame.detectedParameters["bandwidthMhz"] == 40
    assert autoParsedFrame.detectedParameters["sampleRateHz"] == 160.0e6
    assert np.array_equal(
        autoParsedFrame.referenceSignal, autoWaveform.samples
    )
    objectReceivedAnalysis = Analysis(autoWaveform)
    objectReceivedMetrics = objectReceivedAnalysis.Analyze()
    objectReceivedFrame = objectReceivedAnalysis.GetParsedWifiFrame()
    assert objectReceivedFrame is not None
    assert (
        objectReceivedFrame.detectedParameters["frameFormat"]
        == autoWaveform.frameFormat
    )
    assert objectReceivedMetrics["evmDb"] < -200.0

    mimoWaveform = WaveGenWifi(
        frameFormat="HE",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=1,
        sampleRateHz=80.0e6,
        seed=515,
        numTransmitAntennas=2,
        numSpatialStreams=2,
        spatialMapping="dft",
    ).Generate()
    mimoReceived = MimoPaModel(
        numTransmitChains=2
    ).Process(0.20 * mimoWaveform.samples)
    mimoAnalysis = Analysis(
        mimoReceived,
        parseParameters={"sampleRateHz": 80.0e6},
    )
    mimoMetrics = mimoAnalysis.Analyze()
    mimoParsedFrame = mimoAnalysis.GetParsedWifiFrame()
    assert mimoParsedFrame is not None
    assert mimoParsedFrame.detectedParameters["numTransmitAntennas"] == 2
    assert mimoParsedFrame.detectedParameters["numSpatialStreams"] == 2
    assert mimoParsedFrame.detectedParameters["spatialMapping"] == "dft"
    assert np.isfinite(mimoMetrics["evmDb"])
    assert mimoAnalysis.GetLastMimoMetrics() is not None

    explicitAnalysis = Analysis(autoWaveform.samples, autoWaveform)
    try:
        explicitAnalysis.Analyze()
    except ValueError as error:
        assert "measuredSignal is required" in str(error)
    else:
        raise AssertionError(
            "reference-aided Analysis must require a measured signal"
        )


def CheckMseEvmConvergence() -> None:
    """Verify complete separation of ILC diagnostics and RF performance.

    Processing details:
        Algorithm: Prove native ILC records contain only waveform diagnostics
        and measured signals, then pass those outputs to ``Analysis``, verify
        the EVM identity and best-round selection, and validate result files.

    Returns:
        result: None. Assertions enforce metric identities and file content.
    """

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=3,
        oversampling=4,
        seed=91,
    ).Generate()
    referenceSignal = 0.20 * waveform.samples
    resultAnalysis = Analysis(referenceSignal, waveform)
    assert not hasattr(ILCConfig(), "evmMseEvaluator")
    complexGain = 0.72 * np.exp(1j * 0.37)
    gainOnlyOutput = complexGain * referenceSignal
    gainOnlyMetrics = CalculateIterationMetrics(
        1,
        referenceSignal,
        gainOnlyOutput,
        referenceSignal,
    )
    assert gainOnlyMetrics.mse > 1e-4
    assert gainOnlyMetrics.linearCompensatedMse < 1e-25
    assert not hasattr(gainOnlyMetrics, "evmAlignedMse")
    assert not hasattr(gainOnlyMetrics, "evmDb")
    assert np.array_equal(gainOnlyMetrics.inputSignal, referenceSignal)
    assert np.array_equal(gainOnlyMetrics.outputSignal, gainOnlyOutput)

    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        PaModel(modelName="wiener"),
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(
            numIterations=3,
            learningRate=0.25,
            maxAmplitude=1.25,
        ),
    )
    assert len(ilcResult.history) == 3
    for iterationRecord in ilcResult.history:
        assert np.isclose(iterationRecord.mse, iterationRecord.errorRms**2)
        assert not hasattr(iterationRecord, "evmAlignedMse")
        assert not hasattr(iterationRecord, "evmDb")
        assert iterationRecord.inputSignal.shape == referenceSignal.shape
        assert iterationRecord.outputSignal.shape == referenceSignal.shape

    analysisResult = resultAnalysis.AnalyzeIlcHistory(ilcResult.history)
    assert len(analysisResult.history) == 3
    assert analysisResult.bestIteration in (1, 2, 3)
    assert analysisResult.bestInputSignal.shape == referenceSignal.shape
    assert analysisResult.bestOutputSignal.shape == referenceSignal.shape
    for iterationRecord in analysisResult.history:
        assert np.isclose(
            iterationRecord.evmDb,
            10.0 * np.log10(iterationRecord.evmAlignedMse),
        )
        assert np.isclose(
            iterationRecord.evmAlignedMse,
            (iterationRecord.evmPercent / 100.0) ** 2,
        )

    with TemporaryDirectory() as temporaryDirectory:
        outputDirectory = Path(temporaryDirectory)
        csvPath = resultAnalysis.SaveConvergence(
            analysisResult.history, outputDirectory
        )
        figurePath = Draw().SaveConvergenceCurve(
            analysisResult.history, outputDirectory
        )
        csvText = csvPath.read_text(encoding="utf-8-sig")
        assert "mse" in csvText
        assert "linearCompensatedMse" in csvText
        assert "evmAlignedMse" in csvText
        assert "snrDb" in csvText
        assert "aclrWorstDb" in csvText
        assert figurePath.is_file()


def CheckUnknownConfigurationWarnings() -> None:
    """Verify unknown configuration keys warn, disappear, and do not stop work.

    Processing details:
        Algorithm: Exercise constructor mappings, direct keyword overrides,
        live external edits, and update methods across every ChainMap-backed
        public class; capture warnings and confirm recognized settings remain
        operational.

    Returns:
        result: None. Assertions enforce nonfatal unknown-key behavior.
    """

    externalWifiParameters = {
        "mcs": 2,
        "unknownWifiSetting": 100,
    }
    with warnings.catch_warnings(record=True) as capturedWarnings:
        warnings.simplefilter("always")
        wifiGenerator = WaveGenWifi(parameters=externalWifiParameters)
        externalWifiParameters["lateUnknownWifiSetting"] = 200
        wifiGenerator.GetParameters()
        wifiGenerator.UpdateParameters(
            numDataSymbols=2,
            unknownWifiUpdate=300,
        )
        waveform = wifiGenerator.Generate()

        paModel = PaModel(
            parameters={
                "modelName": "wiener",
                "unknownPaSetting": 1,
            },
            unknownPaKeyword=2,
        )
        paModel.UpdateParameters(unknownPaUpdate=3)
        paModel.Process(waveform.samples)

        mimoPaModel = MimoPaModel(
            parameters={
                "numTransmitChains": 1,
                "unknownMimoPaSetting": 1,
            }
        )
        mimoPaModel.Process(waveform.samples)

        powerCalibration = PowerCalibration(
            parameters={
                "loadResistanceOhm": 50.0,
                "unknownPowerSetting": 1,
            }
        )
        powerCalibration.DbmToRms(20.0)

        signalProcessor = SigProc(
            waveform.samples,
            waveform.sampleRateHz,
            parameters={"unknownSignalSetting": 1},
        )
        signalProcessor.Process(waveform.samples)

        wifiParser = ParseWifi(
            parameters={"unknownParserSetting": 1}
        )
        wifiParser.GetParameters()

        resultAnalysis = Analysis(
            waveform.samples,
            waveform,
            parameters={"unknownAnalysisSetting": 1},
        )
        resultAnalysis.Analyze(waveform.samples)

        resultDraw = Draw(
            parameters={"unknownDrawSetting": 1}
        )
        resultDraw.GetParameters()

    warningText = "\n".join(
        str(warningRecord.message)
        for warningRecord in capturedWarnings
    )
    expectedUnknownNames = (
        "unknownWifiSetting",
        "lateUnknownWifiSetting",
        "unknownWifiUpdate",
        "unknownPaSetting",
        "unknownPaKeyword",
        "unknownPaUpdate",
        "unknownMimoPaSetting",
        "unknownPowerSetting",
        "unknownSignalSetting",
        "unknownParserSetting",
        "unknownAnalysisSetting",
        "unknownDrawSetting",
    )
    for unknownName in expectedUnknownNames:
        assert unknownName in warningText
    assert "unknownWifiSetting" not in wifiGenerator.GetParameters()
    assert "unknownPaSetting" not in paModel.GetParameters()
    assert "unknownMimoPaSetting" not in mimoPaModel.GetParameters()
    assert (
        "unknownPowerSetting"
        not in powerCalibration.GetParameters()
    )
    assert "unknownSignalSetting" not in signalProcessor.GetParameters()
    assert "unknownParserSetting" not in wifiParser.GetParameters()
    assert "unknownAnalysisSetting" not in resultAnalysis.GetParameters()
    assert "unknownDrawSetting" not in resultDraw.GetParameters()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            WaveGenWifi(
                parameters={
                    "mcs": 99,
                    "unknownButIgnored": True,
                }
            )
        except ValueError as error:
            assert "MCS" in str(error)
        else:
            raise AssertionError(
                "recognized invalid values must still raise an error"
            )


def RunTests() -> None:
    """Run all project checks and report a compact success message.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    CheckMcsTables()
    CheckFrameFormatAliases()
    CheckFunctionStyle()
    CheckNoGlobalDataVariables()
    CheckModuleResponsibilityBoundaries()
    CheckBenchmarkSeparation()
    CheckFunctionPrincipleCoverage()
    CheckDocumentationMathCompatibility()
    CheckInternalDefaultConfiguration()
    CheckUnknownConfigurationWarnings()
    CheckWifiFormats()
    CheckWifiBandwidths()
    CheckSampleRateConfiguration()
    CheckMimoSpatialStructure()
    CheckMimoPaAndDpd()
    CheckFormatSpecificMcsValidation()
    CheckIdealMetrics()
    CheckSignalProcessingCompensation()
    CheckPowerEvmCurve()
    CheckGuardIntervals()
    CheckIlcImprovement()
    CheckReceiveOnlyWifiAnalysis()
    CheckMseEvmConvergence()
    print("All DPD-ILC project checks passed.")


if __name__ == "__main__":
    RunTests()
