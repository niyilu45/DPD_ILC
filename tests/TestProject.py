"""Self-contained project checks that preserve the requested naming style."""

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import re
import subprocess
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
from inc.lib.Channel import Channel
from inc.lib.DpdGmp import DpdGmp
from inc.lib.DpdIlc import (
    CalculateIterationMetrics,
    FitMimoGmpPredistorter,
    ILCConfig,
    RunComplexGainIlc,
    RunFrequencyDomainIlc,
    RunMimoFrequencyDomainIlc,
)
from inc.lib.Fec import (
    BuildDescriptorLdpcMatrices,
    DecodeDescriptorLdpc,
    EncodeDescriptorLdpc,
)
from inc.utils.Draw import Draw
from inc.utils.FixedPoint import FixedPoint
from inc.lib.PaModel import (
    DohertyConfig,
    DohertyPA,
    GMPConfig,
    MimoPaModel,
    PaModel,
    WienerConfig,
)
from inc.lib.ParseWifi import (
    BuildWifiDescriptorBits,
    DecodeWifiDescriptorBits,
    DescriptorLdpcPhysicalLayout,
    ParseWifi,
)
from inc.utils.SigProc import PowerCalibration, SigProc
from inc.lib.WaveGenWifi import (
    NormalizeFrameFormat,
    WaveGenWifi,
)
from inc.lib.WaveGenTwoTone import WaveGenTwoTone
from inc.lib.TwoToneAnalysis import TwoToneAnalysis


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
    parserSource = (
        projectRoot / "inc" / "lib" / "ParseWifi.py"
    ).read_text(encoding="utf-8")
    fecSource = (
        projectRoot / "inc" / "lib" / "Fec.py"
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
    dpdGmpSource = (
        projectRoot / "inc" / "lib" / "DpdGmp.py"
    ).read_text(encoding="utf-8")

    assert not (projectRoot / "inc" / "SigProcess.py").exists()
    assert not (
        projectRoot / "inc" / "utils" / "ParseWifi.py"
    ).exists()
    for movedModuleName in (
        "Analysis.py",
        "Channel.py",
        "DpdIlc.py",
        "DpdGmp.py",
        "Fec.py",
        "PaModel.py",
        "WaveGenWifi.py",
        "WaveGenTwoTone.py",
        "TwoToneAnalysis.py",
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
    assert "class DpdGmp:" in dpdGmpSource
    assert "from .DpdIlc import BuildFeatureSpecs" in dpdGmpSource
    assert "class PowerCalibration:" not in paSource
    assert "def BuildCsdPhaseMatrix(" in frameProcessorSource
    assert "def BuildCsdPhaseMatrix(" not in waveGeneratorSource
    assert "class MCSInfo:" in metadataSource
    assert "class WifiWaveform:" in metadataSource
    assert "class ParseWifi:" in parserSource
    assert "def EncodeDescriptorLdpc(" in fecSource
    assert "def DecodeDescriptorLdpc(" in fecSource
    assert "def EncodeDescriptorLdpc(" not in parserSource
    assert "def DecodeDescriptorLdpc(" not in parserSource
    assert "from .Fec import" in parserSource
    assert "from .ParseWifi import" in analysisSource
    assert (
        "from .ParseWifi import BuildWifiDescriptorField"
        in waveGeneratorSource
    )
    compatibilityImportCode = (
        "from lib.ParseWifi import ParseWifi; "
        "from lib.Analysis import Analysis; "
        "from lib.Channel import Channel; "
        "from lib.DpdGmp import DpdGmp; "
        "from lib.DpdIlc import RunFrequencyDomainIlc; "
        "from lib.Fec import EncodeDescriptorLdpc; "
        "from lib.WaveGenWifi import WaveGenWifi; "
        "from lib.WaveGenTwoTone import WaveGenTwoTone; "
        "from lib.TwoToneAnalysis import TwoToneAnalysis; "
        "from lib.PaModel import PaModel; "
        "from utils.Draw import Draw"
    )
    compatibilityImportResult = subprocess.run(
        [sys.executable, "-c", compatibilityImportCode],
        cwd=projectRoot / "inc",
        capture_output=True,
        text=True,
        check=False,
    )
    assert compatibilityImportResult.returncode == 0, (
        "top-level lib/utils compatibility imports failed: "
        f"{compatibilityImportResult.stderr}"
    )
    assert "from ..utils.SigProc import" in analysisSource
    assert "from ..utils.SigProc import" in ilcSource
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
        "TwoToneBenchmarkConfig",
        "TwoToneBenchmarkRow",
        "RunTwoToneIlcBenchmark",
        "SaveTwoToneBenchmarkResults",
        "PrintTwoToneBenchmarkResults",
        "PaCharacterizationConfig",
        "PaCharacterizationResult",
        "PaDpdRecommendation",
        "BuildPaDpdRecommendations",
        "RunPaCharacterizationBenchmark",
        "SavePaCharacterizationResults",
        "PrintPaCharacterizationResults",
        "PrintPaDpdRecommendations",
        "DpdGmpBenchmarkConfig",
        "DpdGmpStageResult",
        "DpdGmpImprovementComparison",
        "DpdGmpBenchmarkResult",
        "DpdGmpPaCascade",
        "RunDpdGmpBenchmark",
        "SaveDpdGmpBenchmarkResults",
        "PrintDpdGmpBenchmarkResults",
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
        "B: applicable SISO ILC methods",
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
        "G类：双音IM3/IM5/IM7场景",
        "H类：Wiener/GMP/Doherty PA双音特性",
        "逐PA、逐测试DPD优化建议",
    )
    for sectionTitle in requiredDocumentSections:
        assert sectionTitle in benchmarkDocument, (
            f"missing classified benchmark documentation: {sectionTitle}"
        )
    assert (
        "I类：PA分析驱动的DPD-GMP分阶段性能测试"
        in benchmarkDocument
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


def CheckDocumentationApiConsistency() -> None:
    """Verify runnable Markdown snippets and documented Analysis arguments.

    Processing details:
        Algorithm: Compile every fenced Python example, compare the public
        ``Analysis`` constructor parameter order with its documented
        signature, require synchronization examples to use the explicit
        ``signalProcessingParameters`` argument, and retain one documented
        compatibility note for the legacy nested mapping form.

    Returns:
        result: None. A stale example or signature fails with its document
            path and code-block index.
    """

    projectRoot = GetProjectRoot()
    documentPaths = [projectRoot / "README.md"]
    documentPaths.extend(sorted((projectRoot / "doc").glob("*.md")))
    pythonBlockPattern = re.compile(
        r"```python[ \t]*\r?\n(.*?)```",
        flags=re.DOTALL,
    )
    for documentPath in documentPaths:
        markdownText = documentPath.read_text(encoding="utf-8")
        for blockIndex, pythonBlock in enumerate(
            re.findall(pythonBlockPattern, markdownText),
            start=1,
        ):
            try:
                compile(
                    pythonBlock,
                    f"{documentPath}#python-block-{blockIndex}",
                    "exec",
                )
            except SyntaxError as error:
                raise AssertionError(
                    "invalid Python documentation example at "
                    f"{documentPath} block {blockIndex}: {error}"
                ) from error

    expectedAnalysisParameters = (
        "referenceSignal",
        "waveform",
        "parameters",
        "parseParameters",
        "transmittedSignal",
        "signalProcessingParameters",
        "sampleRateHz",
        "channelBandwidthHz",
        "width",
        "parameterOverrides",
    )
    actualAnalysisParameters = tuple(
        inspect.signature(Analysis).parameters
    )
    assert actualAnalysisParameters == expectedAnalysisParameters

    expectedSignatureText = (
        "Analysis(referenceSignal=None, waveform=None, parameters=None, "
        "parseParameters=None, transmittedSignal=None, "
        "signalProcessingParameters=None, sampleRateHz=None, "
        "channelBandwidthHz=None, width=None, **parameterOverrides)"
    )
    readmeText = (projectRoot / "README.md").read_text(encoding="utf-8")
    analysisDocumentText = (
        projectRoot / "doc" / "Analysis.md"
    ).read_text(encoding="utf-8")
    signalDocumentText = (
        projectRoot / "doc" / "SigProc.md"
    ).read_text(encoding="utf-8")
    assert expectedSignatureText in readmeText
    documentedParameterExpectations = (
        (
            WaveGenWifi,
            (
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "WaveGenWifi(parameters=None, width=None, "
                "**parameterOverrides)"
            ),
        ),
        (
            WaveGenTwoTone,
            (
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "WaveGenTwoTone(parameters=None, width=None, "
                "**parameterOverrides)"
            ),
        ),
        (
            TwoToneAnalysis,
            (
                "waveform",
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "TwoToneAnalysis(waveform, parameters=None, width=None, "
                "**parameterOverrides)"
            ),
        ),
        (
            PowerCalibration,
            (
                "loadResistanceOhm",
                "maximumOutputPowerDbm",
                "paModel",
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "PowerCalibration(loadResistanceOhm=None, "
                "maximumOutputPowerDbm=None, paModel=None, "
                "parameters=None, width=None, **parameterOverrides)"
            ),
        ),
        (
            PaModel,
            (
                "modelName",
                "wienerConfig",
                "gmpConfig",
                "dohertyConfig",
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "PaModel(modelName=None, wienerConfig=None, "
                "gmpConfig=None, dohertyConfig=None, "
                "parameters=None, width=None, "
                "**parameterOverrides)"
            ),
        ),
        (
            Channel,
            (
                "paModel",
                "parameters",
                "width",
                "parameterOverrides",
            ),
            (
                "Channel(paModel=None, parameters=None, width=None, "
                "**parameterOverrides)"
            ),
        ),
    )
    for (
        documentedObject,
        expectedParameterNames,
        documentedSignature,
    ) in documentedParameterExpectations:
        assert tuple(
            inspect.signature(documentedObject).parameters
        ) == expectedParameterNames
        assert documentedSignature in readmeText
    assert tuple(
        inspect.signature(ParseWifi.FindDescriptor).parameters
    ) == (
        "self",
        "receivedSignal",
        "preferredSampleRateHz",
    )
    assert (
        "FindDescriptor(receivedSignal, preferredSampleRateHz=None)"
        in readmeText
    )
    assert tuple(
        inspect.signature(MimoPaModel.ProcessChain).parameters
    ) == ("self", "inputSignal", "chainIndex")
    assert "ProcessChain(inputSignal, chainIndex)" in readmeText
    assert tuple(
        inspect.signature(PowerCalibration.Calibrate).parameters
    ) == ("self", "inputSignal")
    assert "Calibrate(inputSignal)" in readmeText
    assert tuple(
        inspect.signature(Channel.Process).parameters
    ) == ("self", "inputSignal", "outputPowerDbm")
    assert (
        "Process(inputSignal, outputPowerDbm=None)" in readmeText
    )
    assert (
        "chunkSize"
        not in inspect.signature(
            FitMimoGmpPredistorter
        ).parameters
    )
    assert "该接口不包含 `chunkSize`" in readmeText
    assert "signalProcessingParameters=None," in analysisDocumentText
    assert (
        'parameters={"signalProcessingParameters": {...}}'
        in analysisDocumentText
    )
    for documentationText in (
        readmeText,
        analysisDocumentText,
        signalDocumentText,
    ):
        pythonBlocks = re.findall(
            pythonBlockPattern,
            documentationText,
        )
        synchronizationBlocks = [
            pythonBlock
            for pythonBlock in pythonBlocks
            if "signalProcessingParameters" in pythonBlock
        ]
        assert synchronizationBlocks
        for synchronizationBlock in synchronizationBlocks:
            assert "signalProcessingParameters=" in synchronizationBlock


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
        "width": 0,
    }
    wifiGenerator = WaveGenWifi(parameters=externalWifiParameters)
    assert wifiGenerator.frameFormat == "EHT"
    assert wifiGenerator.seed == 7
    assert wifiGenerator.GetParameters()["width"] == 0

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

    externalPaParameters = {
        "modelName": "wiener",
        "width": 0,
    }
    paModel = PaModel(parameters=externalPaParameters)
    assert paModel.modelName == "wiener"
    assert paModel.GetParameters()["width"] == 0
    externalPaParameters["modelName"] = "gmp"
    paModel.Process(np.array([0.1 + 0.0j], dtype=np.complex128))
    assert paModel.modelName == "gmp"
    assert paModel.model.__class__.__name__ == "GMPPA"

    mimoPaModel = MimoPaModel(
        parameters={
            "numTransmitChains": 1,
            "width": 0,
        }
    )
    assert mimoPaModel.GetParameters()["width"] == 0

    wifiParser = ParseWifi(parameters={"width": 0})
    assert wifiParser.GetParameters()["width"] == 0

    analysisParameters = {
        "maxSegmentLength": 1024,
        "width": 0,
    }
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
    assert resultAnalysis.GetParameters()["width"] == 0
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
            width=0,
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
        resultAnalysis = Analysis(
            waveform.samples,
            waveform,
            width=0,
        )
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
        assert (
            len(mimoMetrics["outputPowerDbmPerChain"])
            == transmitCount
        )
        aggregatePowerMilliwatt = sum(
            10.0 ** (powerDbm / 10.0)
            for powerDbm in mimoMetrics["outputPowerDbmPerChain"]
        )
        assert np.isclose(
            idealMetrics["outputPowerDbm"],
            10.0 * np.log10(aggregatePowerMilliwatt),
            atol=1.0e-10,
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
        width=0,
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
        width=0,
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
    resultAnalysis = Analysis(referenceSignal, waveform, width=0)
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

    assert WaveGenWifi(seed=1023).Generate().seed == 1023
    try:
        WaveGenWifi(seed=1024).Generate()
    except ValueError as error:
        assert "2**10 - 1" in str(error)
    else:
        raise AssertionError("a seed wider than 10 bits must be rejected")


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
            "outputPowerDbm",
        }
        assert not hasattr(metrics, "ToDict")
        assert metrics["snrDb"] > 250.0
        assert metrics["evmDb"] < -250.0
        assert metrics["evmPercent"] < 1e-10
        normalizedReferenceRms = float(
            np.sqrt(np.mean(np.abs(resultAnalysis.referenceSignal) ** 2))
        )
        expectedOutputPowerDbm = (
            resultAnalysis.GetParameters()["maximumOutputPowerDbm"]
            + 20.0 * np.log10(normalizedReferenceRms)
        )
        assert np.isclose(
            metrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=1.0e-10,
        )


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
        signalProcessingParameters=signalProcessingParameters,
    )
    metrics = resultAnalysis.Analyze(measuredSignal)
    assert metrics["evmDb"] < -30.0
    expectedOutputPowerDbm = (
        25.0
        + 20.0
        * np.log10(
            abs(expectedComplexGain)
            * np.sqrt(
                np.mean(np.abs(resultAnalysis.referenceSignal) ** 2)
            )
        )
    )
    assert abs(
        metrics["outputPowerDbm"] - expectedOutputPowerDbm
    ) < 0.25
    assert resultAnalysis.GetLastSignalProcessingResult() is not None
    assert (
        resultAnalysis.GetParameters()["signalProcessingParameters"]
        is signalProcessingParameters
    )
    legacyAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "signalProcessingParameters": signalProcessingParameters
        },
    )
    legacyMetrics = legacyAnalysis.Analyze(measuredSignal)
    assert np.isclose(
        legacyMetrics["evmDb"],
        metrics["evmDb"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
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
        width=0,
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
    paModel = PaModel(modelName="wiener", width=0)
    resultAnalysis = Analysis(
        nominalReference,
        waveform,
        loadResistanceOhm=powerCalibration.loadResistanceOhm,
        width=0,
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

    # Power calibration must ignore leading/trailing padding and a long
    # internal off interval. The active bursts deliberately start from an
    # arbitrary 2.7 RMS scale rather than a normalized waveform.
    firstBurst = 2.7 * np.exp(
        1j * 2.0 * np.pi * np.arange(160) / 29.0
    )
    # An eight-sample internal zero crossing is shorter than the configured
    # gap tolerance and must remain inside the active-burst RMS denominator.
    firstBurst[70:78] = 0.0
    secondBurst = 2.7 * np.exp(
        1j * 2.0 * np.pi * np.arange(192) / 31.0
    )
    paddedBurst = np.r_[
        np.zeros(47, dtype=np.complex128),
        firstBurst,
        np.zeros(80, dtype=np.complex128),
        secondBurst,
        np.zeros(53, dtype=np.complex128),
    ]
    floatingPaModel = PaModel(modelName="wiener", width=0)
    floatingCalibration = PowerCalibration(
        paModel=floatingPaModel,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 22.0,
            "calibrationToleranceDb": 0.005,
            "activePowerThresholdDb": -60.0,
            "activeGapToleranceSamples": 16,
            "width": 0,
        }
    )
    floatingCalibratedInput = floatingCalibration.Calibrate(
        paddedBurst
    )
    floatingCalibratedBurst = (
        floatingCalibration.GetLastPaOutput()
    )
    floatingCalibrationMetrics = (
        floatingCalibration.GetLastCalibrationMetrics()
    )
    assert (
        max(
            abs(errorDb)
            for errorDb in floatingCalibrationMetrics[
                "errorDbPerChain"
            ]
        )
        <= floatingCalibration.GetParameters()[
            "calibrationToleranceDb"
        ]
    )
    assert "driveScale" not in floatingCalibrationMetrics
    assert np.array_equal(
        floatingCalibration.GetLastPaInput(),
        floatingCalibratedInput,
    )
    firstCalibrationIterationCount = floatingCalibrationMetrics[
        "iterationCount"
    ]
    repeatedCalibratedInput = floatingCalibration.Calibrate(
        paddedBurst
    )
    repeatedCalibrationMetrics = (
        floatingCalibration.GetLastCalibrationMetrics()
    )
    assert np.array_equal(
        repeatedCalibratedInput,
        floatingCalibration.GetLastPaInput(),
    )
    assert (
        repeatedCalibrationMetrics["iterationCount"]
        <= firstCalibrationIterationCount
    )
    floatingActiveMask = floatingCalibration.FindActiveSampleMask(
        floatingCalibratedInput
    )
    assert np.count_nonzero(floatingActiveMask) == 352
    floatingActiveRms = (
        floatingCalibration.CalculateActiveRmsPerChain(
            floatingCalibratedBurst
        )[0]
    )
    assert np.isclose(
        floatingCalibration.NormalizedRmsToOutputPowerDbm(
            floatingActiveRms
        ),
        22.0,
        atol=0.005,
    )
    assert (
        np.sqrt(np.mean(np.abs(floatingCalibratedBurst) ** 2))
        < floatingActiveRms
    )

    mimoPaModel = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "width": 0,
        }
    )
    mimoCalibration = PowerCalibration(
        paModel=mimoPaModel,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbmPerChain": (18.0, 22.0),
            "calibrationToleranceDb": 0.005,
            "activePowerThresholdDb": -60.0,
            "activeGapToleranceSamples": 16,
            "width": 0,
        },
    )
    mimoCalibration.Calibrate(
        np.column_stack((paddedBurst, 0.17 * paddedBurst))
    )
    mimoCalibratedBurst = mimoCalibration.GetLastPaOutput()
    mimoActiveRms = (
        mimoCalibration.CalculateActiveRmsPerChain(
            mimoCalibratedBurst
        )
    )
    assert np.allclose(
        tuple(
            mimoCalibration.NormalizedRmsToOutputPowerDbm(
                chainRms
            )
            for chainRms in mimoActiveRms
        ),
        (18.0, 22.0),
        atol=0.005,
    )

    fixedPaModel = PaModel(modelName="wiener", width=16)
    fixedCalibration = PowerCalibration(
        paModel=fixedPaModel,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 22.0,
            "calibrationToleranceDb": 0.005,
            "activePowerThresholdDb": -60.0,
            "activeGapToleranceSamples": 16,
            "width": 16,
        }
    )
    fixedInputBurst = FixedPoint(16).EncodeComplex(
        paddedBurst / 3.0
    )
    fixedCalibratedInput = fixedCalibration.Calibrate(
        fixedInputBurst
    )
    fixedCalibratedBurst = fixedCalibration.GetLastPaOutput()
    assert np.allclose(
        fixedCalibratedInput.real,
        np.rint(fixedCalibratedInput.real),
    )
    assert np.allclose(
        fixedCalibratedInput.imag,
        np.rint(fixedCalibratedInput.imag),
    )
    decodedFixedBurst = FixedPoint(16).DecodeComplex(
        fixedCalibratedBurst
    )
    fixedActiveRms = fixedCalibration.CalculateActiveRmsPerChain(
        decodedFixedBurst
    )[0]
    assert abs(
        fixedCalibration.NormalizedRmsToOutputPowerDbm(
            fixedActiveRms
        )
        - 22.0
    ) < 0.01
    fixedBurstMetrics = Analysis(
        fixedCalibratedBurst,
        transmittedSignal=fixedCalibratedBurst,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "activePowerThresholdDb": -60.0,
            "activeGapToleranceSamples": 16,
            "width": 16,
        },
    ).Analyze()
    assert abs(fixedBurstMetrics["outputPowerDbm"] - 22.0) < 0.01

    voltageCalibratedBurst = (
        floatingCalibration.ScaleSignalToOutputPower(
            paddedBurst,
            22.0,
        )
    )
    voltageActiveRms = (
        floatingCalibration.CalculateActiveRmsPerChain(
            voltageCalibratedBurst
        )[0]
    )
    assert np.isclose(
        floatingCalibration.RmsToDbm(voltageActiveRms),
        22.0,
        atol=1.0e-10,
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
    resultAnalysis = Analysis(
        referenceSignal, waveform, parameters={"width": 0}
    )
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


def CheckDohertyPaModel() -> None:
    """Verify branch turn-on, combining, facade selection, and validation.

    Processing details:
        Algorithm: Compare low-envelope output with the continuously active
        carrier branch, require the peaking branch to alter high-envelope
        samples, verify analytic small-signal gain against a numerical probe,
        exercise mixed Doherty/GMP MIMO configuration, and reject invalid
        branch, turn-on, transition, delay, and load-modulation settings.

    Returns:
        result: None. Assertions enforce the behavioral Doherty contract.
    """

    dohertyConfig = DohertyConfig(
        carrierModelName="wiener",
        peakingModelName="gmp",
        carrierWienerConfig=WienerConfig(
            linearTaps=(1.0 + 0.0j,),
            linearGain=0.9,
            saturationAmplitude=1.0,
            rappSmoothness=3.0,
            ampmCoefficient=0.0,
        ),
        peakingGmpConfig=GMPConfig(
            nonlinearOrders=(1, 3),
            memoryDepth=1,
            crossMemoryDepth=0,
        ),
        carrierInputGain=1.0,
        peakingInputGain=0.8,
        peakingTurnOnAmplitude=0.40,
        peakingTransitionWidth=0.20,
        carrierCombineCoefficient=1.0 + 0.0j,
        peakingCombineCoefficient=0.45 * np.exp(1j * 0.1),
        peakingDelaySamples=2,
        loadModulationStrength=0.12,
    )
    dohertyPa = DohertyPA(dohertyConfig)
    lowInput = (
        0.05
        * np.exp(
            1j
            * 2.0
            * np.pi
            * np.arange(128, dtype=float)
            / 17.0
        )
    )
    lowOutput = dohertyPa.Process(lowInput)
    expectedCarrierOnly = (
        dohertyConfig.carrierCombineCoefficient
        * dohertyPa.carrierModel.Process(
            dohertyConfig.carrierInputGain * lowInput
        )
    )
    assert np.allclose(lowOutput, expectedCarrierOnly)
    highInput = 0.80 * lowInput / 0.05
    highOutput = dohertyPa.Process(highInput)
    carrierOnlyHigh = (
        dohertyConfig.carrierCombineCoefficient
        * dohertyPa.carrierModel.Process(
            dohertyConfig.carrierInputGain * highInput
        )
    )
    assert not np.allclose(highOutput, carrierOnlyHigh)
    tinyInput = np.full(
        64, 1.0e-7 + 0.0j, dtype=np.complex128
    )
    numericalSmallSignalGain = np.mean(
        dohertyPa.Process(tinyInput) / tinyInput
    )
    assert abs(
        numericalSmallSignalGain - dohertyPa.SmallSignalGain()
    ) < 1.0e-8

    facadePa = PaModel(
        parameters={
            "modelName": "doherty",
            "dohertyConfig": dohertyConfig,
            "width": 0,
        }
    )
    assert facadePa.modelName == "doherty"
    assert np.allclose(facadePa.Process(highInput), highOutput)
    mixedMimoPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {
                    "modelName": "doherty",
                    "dohertyConfig": dohertyConfig,
                },
                {"modelName": "gmp"},
            ),
            "width": 0,
        }
    )
    mixedOutput = mixedMimoPa.Process(
        np.column_stack((highInput, highInput))
    )
    assert mixedOutput.shape == (highInput.size, 2)
    assert not np.allclose(
        mixedOutput[:, 0], mixedOutput[:, 1]
    )

    invalidDohertyConfigs = (
        {"carrierModelName": "memoryPolynomial"},
        {"peakingInputGain": 0.0},
        {"peakingTurnOnAmplitude": 0.0},
        {"peakingTransitionWidth": 0.0},
        {"carrierCombineCoefficient": 0.0 + 0.0j},
        {"peakingDelaySamples": -1},
        {"loadModulationStrength": -0.1},
    )
    for invalidOverrides in invalidDohertyConfigs:
        try:
            DohertyPA(DohertyConfig(**invalidOverrides))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                "invalid Doherty configuration accepted: "
                f"{invalidOverrides!r}"
            )


def CheckIlcFeedbackSynchronization() -> None:
    """Verify that ILC aligns feedback before metrics and waveform updates.

    Processing details:
        Algorithm: Pass an ideal Wi-Fi waveform through a synthetic capture
        path with leading delay, carrier offset, common complex gain, and a
        longer record; run frequency-domain ILC; then verify that every native
        history record is in the synchronized gain-normalized reference domain
        and exports the estimated impairment values through Analysis.

    Returns:
        result: None. Assertions prove synchronization is inside the ILC loop
            rather than deferred to post-run performance reporting.
    """

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=2,
        numDataSymbols=8,
        sampleRateHz=80.0e6,
        seed=177,
        width=0,
    ).Generate()
    referenceSignal = waveform.samples
    delaySamples = 19
    carrierFrequencyOffsetHz = 5.0e3
    complexGain = 0.61 * np.exp(1j * 0.42)

    class ImpairedIdentityPa:
        """Add deterministic capture impairments to an identity PA."""

        def __init__(
            self,
            sampleRateHz: float,
            delaySamples: int,
            carrierFrequencyOffsetHz: float,
            complexGain: complex,
        ) -> None:
            """Store deterministic impairment values for repeated captures.

            Processing details:
                Algorithm: Preserve the caller's physical sample rate, signed
                phase-ramp frequency, leading zero count, and common gain.

            Args:
                sampleRateHz: Complex sample rate in samples per second.
                delaySamples: Number of leading zero samples.
                carrierFrequencyOffsetHz: Applied carrier offset in hertz.
                complexGain: Applied common magnitude and phase.

            Returns:
                result: None. The repeatable identity test plant is ready.
            """

            self.sampleRateHz = sampleRateHz
            self.delaySamples = delaySamples
            self.carrierFrequencyOffsetHz = (
                carrierFrequencyOffsetHz
            )
            self.complexGain = complexGain

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Return a longer delayed, frequency-shifted, gained capture.

            Processing details:
                Algorithm: Prefix zeros, preserve every input sample, and
                apply the same absolute-index carrier phasor on every call.

            Args:
                inputSignal: One-dimensional complex identity-plant input.

            Returns:
                result: Impaired capture with ``delaySamples`` extra samples.
            """

            complexInput = np.asarray(
                inputSignal, dtype=np.complex128
            ).reshape(-1)
            delayedSignal = np.r_[
                np.zeros(
                    self.delaySamples,
                    dtype=np.complex128,
                ),
                complexInput,
            ]
            sampleIndices = np.arange(delayedSignal.size, dtype=float)
            carrierPhasor = np.exp(
                1j
                * 2.0
                * np.pi
                * self.carrierFrequencyOffsetHz
                * sampleIndices
                / self.sampleRateHz
            )
            return self.complexGain * delayedSignal * carrierPhasor

    impairedPa = ImpairedIdentityPa(
        waveform.sampleRateHz,
        delaySamples,
        carrierFrequencyOffsetHz,
        complexGain,
    )
    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        impairedPa,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(
            numIterations=2,
            learningRate=0.2,
            maxAmplitude=2.5,
            feedbackSynchronizationParameters={
                "maxIntegerDelaySamples": 100,
            },
        ),
    )
    assert ilcResult.outputSignal.size == (
        referenceSignal.size + delaySamples
    )
    for iterationRecord in ilcResult.history:
        assert iterationRecord.outputSignal.shape == referenceSignal.shape
        assert iterationRecord.integerDelaySamples == delaySamples
        assert abs(
            iterationRecord.carrierFrequencyOffsetHz
            - carrierFrequencyOffsetHz
        ) < 100.0
        assert abs(
            iterationRecord.feedbackComplexGain - complexGain
        ) < 0.02
        assert iterationRecord.linearCompensatedNmseDb < -35.0

    analyzedResult = Analysis(
        referenceSignal,
        waveform,
        parameters={"width": 0},
    ).AnalyzeIlcHistory(ilcResult.history)
    serializedIteration = analyzedResult.history[0].ToDict()
    assert (
        serializedIteration["feedbackIntegerDelaySamples"]
        == delaySamples
    )
    assert abs(
        serializedIteration["feedbackCarrierFrequencyOffsetHz"]
        - carrierFrequencyOffsetHz
    ) < 100.0
    assert "feedbackComplexGainMagnitudeDb" in serializedIteration


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
    fecMessageBits = np.zeros(55, dtype=np.uint8)
    fecMessageBits[::4] = 1
    fecCodeword = EncodeDescriptorLdpc(fecMessageBits)
    parityCheckMatrix, messageMatrix = BuildDescriptorLdpcMatrices()
    assert parityCheckMatrix.shape == (35, 90)
    assert messageMatrix.shape == (35, 55)
    assert not np.any(
        np.mod(
            parityCheckMatrix.astype(np.int64)
            @ fecCodeword.astype(np.int64),
            2,
        )
    )
    fecSoftCodeword = 1.0 - 2.0 * fecCodeword.astype(float)
    fecSoftCodeword[[2, 17, 44, 71]] *= -1.0
    assert np.array_equal(
        DecodeDescriptorLdpc(fecSoftCodeword),
        fecMessageBits,
    )

    descriptorBits = BuildWifiDescriptorBits(
        "EHT",
        20,
        7,
        10,
        0.8,
        101,
        1,
        1,
        "direct",
        True,
    )
    (
        _,
        _,
        codePhysicalPositions,
        _,
    ) = DescriptorLdpcPhysicalLayout()
    corruptedDescriptorBits = descriptorBits.copy()
    corruptedDescriptorBits[
        codePhysicalPositions[[2, 17, 44, 71]]
    ] ^= 1
    correctedDescriptor = DecodeWifiDescriptorBits(
        corruptedDescriptorBits
    )
    assert correctedDescriptor["seed"] == 101

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
            width=0,
        ).Generate()
        assert waveform.seed == 501 + formatIndex
        assert waveform.cyclicShiftEnabled
        measuredSignal = PaModel(
            modelName="wiener",
            width=0,
        ).Process(
            0.22 * waveform.samples
        )
        referenceMetrics = Analysis(
            waveform.samples, waveform, width=0
        ).Analyze(measuredSignal)
        nullReferenceAnalysis = Analysis(
            referenceSignal=None,
            waveform=waveform,
            width=0,
        )
        nullReferenceMetrics = nullReferenceAnalysis.Analyze(
            measuredSignal
        )
        omittedReferenceAnalysis = Analysis(
            waveform=waveform,
            width=0,
        )
        omittedReferenceMetrics = omittedReferenceAnalysis.Analyze(
            measuredSignal
        )
        assert (
            nullReferenceAnalysis.GetAnalysisMode()
            == "explicitReference"
        )
        assert np.array_equal(
            nullReferenceAnalysis.referenceSignal,
            waveform.samples,
        )
        assert np.allclose(
            np.asarray(tuple(referenceMetrics.values())),
            np.asarray(tuple(nullReferenceMetrics.values())),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert np.allclose(
            np.asarray(tuple(referenceMetrics.values())),
            np.asarray(tuple(omittedReferenceMetrics.values())),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        try:
            Analysis(
                None,
                transmittedSignal=waveform.samples,
                width=0,
            )
        except ValueError as error:
            assert "received signal cannot be None" in str(error)
        else:
            raise AssertionError(
                "transmit-assisted mode requires a received signal"
            )
        try:
            Analysis(None, width=0)
        except ValueError as error:
            assert "received signal cannot be None" in str(error)
        else:
            raise AssertionError(
                "blind mode requires a received signal"
            )
        leadingSamples = 13 + formatIndex
        receiveCapture = np.r_[
            np.zeros(leadingSamples, dtype=np.complex128),
            measuredSignal,
            np.zeros(7, dtype=np.complex128),
        ]
        receiveAnalysis = Analysis(
            receiveCapture,
            parseParameters={"sampleRateHz": 80.0e6},
            width=0,
        )
        receiveMetrics = receiveAnalysis.Analyze()
        parsedFrame = receiveAnalysis.GetParsedWifiFrame()
        assert receiveAnalysis.GetAnalysisMode() == "blind"
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
            sampleRateHz=waveform.sampleRateHz,
            channelBandwidthHz=waveform.bandwidthHz,
            width=0,
        )
        assistedMetrics = assistedAnalysis.Analyze()
        assistedFrame = assistedAnalysis.GetParsedWifiFrame()
        assistedOverlap = assistedAnalysis.GetSignalOverlapResult()
        assert assistedFrame is None
        assert assistedAnalysis.GetAnalysisMode() == "transmitAssisted"
        assert assistedOverlap is not None
        assert assistedOverlap.receivedStartSample == leadingSamples
        assert assistedOverlap.referenceStartSample == 0
        assert assistedOverlap.confidence > 0.90
        assert np.isfinite(assistedMetrics["evmDb"])
        assert np.isfinite(assistedMetrics["aclrWorstDb"])
        if formatIndex == 0:
            with warnings.catch_warnings(record=True) as warningRecords:
                warnings.simplefilter("always")
                compatibleAssistedAnalysis = Analysis(
                    receiveCapture,
                    transmittedSignal=waveform.samples,
                    parseParameters={
                        "sampleRateHz": waveform.sampleRateHz,
                        "channelBandwidthHz": waveform.bandwidthHz,
                        "maximumPacketOffsetSamples": 128,
                    },
                    width=0,
                )
            compatibleAssistedMetrics = (
                compatibleAssistedAnalysis.Analyze()
            )
            assert (
                compatibleAssistedAnalysis.GetAnalysisMode()
                == "transmitAssisted"
            )
            assert (
                compatibleAssistedAnalysis.GetParsedWifiFrame() is None
            )
            assert (
                compatibleAssistedAnalysis.sampleRateHz
                == waveform.sampleRateHz
            )
            assert (
                compatibleAssistedAnalysis.channelBandwidthHz
                == waveform.bandwidthHz
            )
            assert np.isfinite(
                compatibleAssistedMetrics["aclrWorstDb"]
            )
            assert any(
                "maximumPacketOffsetSamples" in str(warningRecord.message)
                for warningRecord in warningRecords
            )
            precedenceAnalysis = Analysis(
                receiveCapture,
                transmittedSignal=waveform.samples,
                parseParameters={"sampleRateHz": 60.0e6},
                sampleRateHz=waveform.sampleRateHz,
                channelBandwidthHz=waveform.bandwidthHz,
                width=0,
            )
            assert precedenceAnalysis.sampleRateHz == waveform.sampleRateHz
            objectAssistedAnalysis = Analysis(
                receiveCapture,
                transmittedSignal=waveform,
                width=0,
            )
            objectAssistedMetrics = objectAssistedAnalysis.Analyze()
            objectAssistedFrame = (
                objectAssistedAnalysis.GetParsedWifiFrame()
            )
            assert objectAssistedFrame is None
            assert (
                objectAssistedAnalysis.GetAnalysisMode()
                == "transmitAssisted"
            )
            assert (
                objectAssistedAnalysis.GetSignalOverlapResult()
                is not None
            )
            assert (
                objectAssistedAnalysis.GetSignalOverlapResult()
                .receivedStartSample
                == leadingSamples
            )
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
                sampleRateHz=waveform.sampleRateHz,
                channelBandwidthHz=waveform.bandwidthHz,
                width=0,
            )
            objectReceiveArrayTransmitMetrics = (
                objectReceiveArrayTransmitAnalysis.Analyze()
            )
            objectReceiveObjectTransmitAnalysis = Analysis(
                receivedWaveform,
                transmittedSignal=waveform,
                width=0,
            )
            objectReceiveObjectTransmitMetrics = (
                objectReceiveObjectTransmitAnalysis.Analyze()
            )
            assert np.isfinite(
                objectReceiveArrayTransmitMetrics["evmDb"]
            )
            assert np.allclose(
                np.asarray(tuple(referenceMetrics.values())),
                np.asarray(
                    tuple(objectReceiveObjectTransmitMetrics.values())
                ),
                rtol=1.0e-10,
                atol=1.0e-10,
            )

        typicalDriveScale = 10.0 ** (-5.0 / 20.0)
        gmpMeasuredSignal = PaModel(
            modelName="gmp",
            width=0,
        ).Process(
            typicalDriveScale * waveform.samples
        )
        gmpReferenceMetrics = Analysis(
            waveform.samples,
            waveform,
            width=0,
        ).Analyze(gmpMeasuredSignal)
        gmpReceiveAnalysis = Analysis(gmpMeasuredSignal, width=0)
        gmpReceiveMetrics = gmpReceiveAnalysis.Analyze()
        gmpParsedFrame = gmpReceiveAnalysis.GetParsedWifiFrame()
        assert gmpParsedFrame is not None
        assert (
            gmpParsedFrame.detectedParameters["frameFormat"]
            == frameFormat
        )
        assert gmpParsedFrame.detectedParameters["mcs"] == maximumMcs
        assert gmpParsedFrame.detectedParameters["seed"] == (
            501 + formatIndex
        )
        assert np.allclose(
            np.asarray(tuple(gmpReferenceMetrics.values())),
            np.asarray(tuple(gmpReceiveMetrics.values())),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    smallestSisoWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=10,
        sampleRateHz=80.0e6,
        seed=101,
        width=0,
    ).Generate()
    smallestSisoDriveScale = 10.0 ** (-5.0 / 20.0)
    smallestSisoPaOutput = PaModel(
        modelName="gmp",
        width=0,
    ).Process(
        smallestSisoDriveScale * smallestSisoWaveform.samples
    )
    smallestSisoReferenceMetrics = Analysis(
        smallestSisoDriveScale * smallestSisoWaveform.samples,
        smallestSisoWaveform,
        width=0,
    ).Analyze(smallestSisoPaOutput)
    smallestSisoReceiveAnalysis = Analysis(
        smallestSisoPaOutput,
        width=0,
    )
    smallestSisoReceiveMetrics = (
        smallestSisoReceiveAnalysis.Analyze()
    )
    smallestSisoParsedFrame = (
        smallestSisoReceiveAnalysis.GetParsedWifiFrame()
    )
    assert smallestSisoParsedFrame is not None
    assert smallestSisoParsedFrame.detectedParameters["seed"] == 101
    assert np.allclose(
        np.asarray(tuple(smallestSisoReferenceMetrics.values())),
        np.asarray(tuple(smallestSisoReceiveMetrics.values())),
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
        width=0,
    ).Generate()
    autoParsedFrame = ParseWifi(
        parameters={"width": 0}
    ).Parse(autoWaveform.samples)
    assert autoParsedFrame.detectedParameters["bandwidthMhz"] == 40
    assert autoParsedFrame.detectedParameters["sampleRateHz"] == 160.0e6
    assert np.array_equal(
        autoParsedFrame.referenceSignal, autoWaveform.samples
    )
    overlapParser = ParseWifi(parameters={"width": 0})
    assert (
        overlapParser.GetParameters()["maximumPacketOffsetSamples"]
        == 2000
    )
    cropStartSample = 37
    cropStopSample = autoWaveform.samples.shape[0] - 29
    croppedReceive = (
        1.7
        * autoWaveform.samples[cropStartSample:cropStopSample]
    )
    (
        receiveOverlapStart,
        transmitOverlapStart,
        overlapLength,
        overlapConfidence,
    ) = overlapParser.EstimateSignalOverlap(
        croppedReceive,
        autoWaveform.samples,
    )
    assert receiveOverlapStart == 0
    assert transmitOverlapStart == cropStartSample
    assert overlapLength == croppedReceive.shape[0]
    assert overlapConfidence > 0.999
    croppedAnalysis = Analysis(
        croppedReceive,
        transmittedSignal=autoWaveform,
        width=0,
    )
    croppedMetrics = croppedAnalysis.Analyze()
    assert np.isfinite(croppedMetrics["evmPercent"])
    croppedPaReceive = PaModel(
        modelName="gmp",
        width=0,
    ).Process(
        0.25
        * autoWaveform.samples[
            cropStartSample:cropStopSample
        ]
    )
    croppedPaAnalysis = Analysis(
        croppedPaReceive,
        transmittedSignal=autoWaveform,
        width=0,
    )
    croppedPaMetrics = croppedPaAnalysis.Analyze()
    croppedPaFrame = croppedPaAnalysis.GetParsedWifiFrame()
    assert croppedPaFrame is None
    assert croppedPaAnalysis.GetAnalysisMode() == "transmitAssisted"
    assert np.isfinite(croppedPaMetrics["evmPercent"])

    transmitPaddingBefore = 41
    paddedTransmit = np.r_[
        np.zeros(
            transmitPaddingBefore,
            dtype=np.complex128,
        ),
        autoWaveform.samples,
        np.zeros(53, dtype=np.complex128),
    ]
    effectiveReceive = 1.7 * autoWaveform.samples
    paddedTransmitAnalysis = Analysis(
        effectiveReceive,
        transmittedSignal=paddedTransmit,
        sampleRateHz=autoWaveform.sampleRateHz,
        channelBandwidthHz=autoWaveform.bandwidthHz,
        width=0,
    )
    paddedTransmitOverlap = (
        paddedTransmitAnalysis.GetSignalOverlapResult()
    )
    paddedTransmitMetrics = paddedTransmitAnalysis.Analyze()
    assert paddedTransmitAnalysis.GetParsedWifiFrame() is None
    assert paddedTransmitOverlap is not None
    assert paddedTransmitOverlap.receivedStartSample == 0
    assert (
        paddedTransmitOverlap.referenceStartSample
        == transmitPaddingBefore
    )
    assert paddedTransmitOverlap.confidence > 0.999
    assert paddedTransmitMetrics["evmPercent"] < 1.0e-8

    # Raw NumPy-assisted analysis must work even when the transmit reference
    # no longer contains the protected descriptor or complete Wi-Fi fields.
    dataOnlyTransmit = autoWaveform.samples[
        autoWaveform.fieldSlices[autoWaveform.dataFieldName]
    ]
    transmitVariants = (
        autoWaveform.samples[500:],
        autoWaveform.samples[:-300],
        np.r_[
            np.zeros(800, dtype=np.complex128),
            autoWaveform.samples,
        ],
        autoWaveform.samples[1000:9000],
        dataOnlyTransmit,
    )
    for transmitVariant in transmitVariants:
        assistedReceive = np.r_[
            np.zeros(23, dtype=np.complex128),
            0.73 * np.exp(1j * 0.31) * transmitVariant,
            np.zeros(11, dtype=np.complex128),
        ]
        waveformAssistedAnalysis = Analysis(
            assistedReceive,
            transmittedSignal=transmitVariant,
            width=0,
        )
        waveformAssistedMetrics = waveformAssistedAnalysis.Analyze()
        waveformOverlap = (
            waveformAssistedAnalysis.GetSignalOverlapResult()
        )
        assert (
            waveformAssistedAnalysis.GetAnalysisMode()
            == "transmitAssisted"
        )
        assert waveformAssistedAnalysis.GetParsedWifiFrame() is None
        assert waveformOverlap is not None
        assert (
            waveformOverlap.receivedStartSample
            - waveformOverlap.referenceStartSample
            == 23
        )
        assert waveformOverlap.confidence > 0.999
        assert waveformAssistedMetrics["evmPercent"] < 1.0e-8
        assert waveformAssistedMetrics["snrDb"] > 150.0
        assert np.isnan(waveformAssistedMetrics["aclrWorstDb"])

    objectReceivedAnalysis = Analysis(autoWaveform, width=0)
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
        width=0,
    ).Generate()
    mimoReceived = MimoPaModel(
        numTransmitChains=2,
        width=0,
    ).Process(0.20 * mimoWaveform.samples)
    mimoAnalysis = Analysis(
        mimoReceived,
        parseParameters={"sampleRateHz": 80.0e6},
        width=0,
    )
    mimoMetrics = mimoAnalysis.Analyze()
    mimoParsedFrame = mimoAnalysis.GetParsedWifiFrame()
    assert mimoParsedFrame is not None
    assert mimoParsedFrame.detectedParameters["numTransmitAntennas"] == 2
    assert mimoParsedFrame.detectedParameters["numSpatialStreams"] == 2
    assert mimoParsedFrame.detectedParameters["spatialMapping"] == "dft"
    assert np.isfinite(mimoMetrics["evmDb"])
    assert mimoAnalysis.GetLastMimoMetrics() is not None

    explicitAnalysis = Analysis(
        autoWaveform.samples,
        autoWaveform,
        width=0,
    )
    assert explicitAnalysis.GetAnalysisMode() == "explicitReference"
    assert explicitAnalysis.GetParsedWifiFrame() is None
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
        width=0,
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
        PaModel(modelName="wiener", width=0),
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
            10.0
            * np.log10(
                max(
                    iterationRecord.evmAlignedMse,
                    np.finfo(float).tiny,
                )
            ),
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
        assert "feedbackIntegerDelaySamples" in csvText
        assert "feedbackCarrierFrequencyOffsetHz" in csvText
        assert "feedbackComplexGainMagnitudeDb" in csvText
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

        channel = Channel(
            parameters={
                "width": 0,
                "unknownChannelSetting": 1,
            }
        )
        channel.UpdateParameters(unknownChannelUpdate=2)
        channel.ProcessPaOutput(waveform.samples)

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
        "unknownChannelSetting",
        "unknownChannelUpdate",
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
    assert "unknownChannelSetting" not in channel.GetParameters()
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


def CheckFixedPointInterfaces() -> None:
    """Verify shared float and fixed I/Q boundaries for the three main APIs.

    Processing details:
        Algorithm: Check exact raw-code encoding, saturation, and decoding,
        require an unchanged complex128 container type in both modes, exercise
        14- and 16-bit generators, and pass raw codes through PA and Analysis.

    Returns:
        result: None. Assertions identify interface-format regressions.
    """

    floatingFormat = FixedPoint(width=0)
    fixedFormat = FixedPoint(width=3)
    inputSignal = np.array(
        [0.13 + 0.37j, 2.0 - 2.0j, -0.20 + 0.0j],
        dtype=np.complex128,
    )
    floatingSignal = floatingFormat.QuantizeComplex(inputSignal)
    fixedSignal = fixedFormat.QuantizeComplex(inputSignal)
    expectedFixedSignal = np.array(
        [1.0 + 1.0j, 3.0 - 4.0j, -1.0 + 0.0j],
        dtype=np.complex128,
    )
    expectedDecodedSignal = np.array(
        [0.25 + 0.25j, 0.75 - 1.0j, -0.25 + 0.0j],
        dtype=np.complex128,
    )
    assert floatingFormat.IsFloatingPoint()
    assert not fixedFormat.IsFloatingPoint()
    assert floatingFormat.GetFormatInfo()["mode"] == "floating"
    assert fixedFormat.GetFormatInfo()["fractionalBits"] == 2
    assert floatingSignal.dtype == np.complex128
    assert fixedSignal.dtype == np.complex128
    assert floatingSignal.shape == fixedSignal.shape == inputSignal.shape
    assert np.array_equal(floatingSignal, inputSignal)
    assert np.array_equal(fixedSignal, expectedFixedSignal)
    assert np.array_equal(
        fixedFormat.DecodeComplex(fixedSignal),
        expectedDecodedSignal,
    )
    assert np.array_equal(
        fixedFormat.QuantizeCodes(
            np.array([1.4 - 5.2j, -9.0 + 2.6j])
        ),
        np.array([1.0 - 4.0j, -4.0 + 3.0j]),
    )
    assert fixedFormat.GetFormatInfo()["minimumCode"] == -4.0
    assert fixedFormat.GetFormatInfo()["maximumCode"] == 3.0

    defaultGenerator = WaveGenWifi(
        bandwidthMhz=20,
        numDataSymbols=1,
        sampleRateHz=80.0e6,
    )
    floatingGenerator = WaveGenWifi(
        parameters={"width": 0},
        bandwidthMhz=20,
        numDataSymbols=1,
        sampleRateHz=80.0e6,
    )
    defaultWaveform = defaultGenerator.Generate()
    floatingWaveform = floatingGenerator.Generate()
    fourteenBitWaveform = WaveGenWifi(
        parameters={
            "width": 14,
            "bandwidthMhz": 20,
            "numDataSymbols": 1,
            "sampleRateHz": 80.0e6,
        }
    ).Generate()
    assert defaultGenerator.width == 16
    assert floatingGenerator.width == 0
    assert defaultWaveform.samples.dtype == np.complex128
    assert floatingWaveform.samples.dtype == np.complex128
    assert defaultWaveform.samples.shape == floatingWaveform.samples.shape
    assert np.array_equal(
        defaultWaveform.samples.real,
        np.rint(defaultWaveform.samples.real),
    )
    assert np.array_equal(
        defaultWaveform.samples.imag,
        np.rint(defaultWaveform.samples.imag),
    )
    assert np.max(defaultWaveform.samples.real) <= 32767.0
    assert np.min(defaultWaveform.samples.real) >= -32768.0
    assert np.max(fourteenBitWaveform.samples.real) <= 8191.0
    assert np.min(fourteenBitWaveform.samples.real) >= -8192.0

    paInput = 0.5 * floatingWaveform.samples
    fixedPaInput = 0.5 * defaultWaveform.samples
    floatingPaOutput = PaModel(
        parameters={"width": 0}
    ).Process(paInput)
    fixedPaOutput = PaModel(
        parameters={"width": 16}
    ).Process(fixedPaInput)
    assert floatingPaOutput.dtype == np.complex128
    assert fixedPaOutput.dtype == np.complex128
    assert floatingPaOutput.shape == fixedPaOutput.shape
    assert np.array_equal(
        fixedPaOutput.real, np.rint(fixedPaOutput.real)
    )
    assert np.array_equal(
        fixedPaOutput.imag, np.rint(fixedPaOutput.imag)
    )

    floatingMetrics = Analysis(
        floatingWaveform.samples,
        floatingWaveform,
        parameters={"width": 0},
    ).Analyze(floatingWaveform.samples)
    fixedAnalysis = Analysis(
        defaultWaveform.samples,
        defaultWaveform,
        parameters={"width": 16},
    )
    fixedMetrics = fixedAnalysis.Analyze(defaultWaveform.samples)
    assert fixedAnalysis.GetParameters()["width"] == 16
    assert fixedAnalysis.width == 16
    assert set(floatingMetrics) == set(fixedMetrics)
    assert isinstance(floatingMetrics, dict)
    assert isinstance(fixedMetrics, dict)
    assert np.isfinite(floatingMetrics["evmDb"])
    assert np.isfinite(fixedMetrics["evmDb"])

    for invalidWidth in (-1, 1.5, True, 54):
        try:
            FixedPoint(invalidWidth)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"invalid fixed-point width accepted: {invalidWidth!r}"
            )


def CheckChannelModel() -> None:
    """Verify both sample modes, feedback impairments, noise, and fixed point.

    Processing details:
        Algorithm: Prove forward sampling bypasses embedded receiver defects,
        exercise deterministic feedback gain/FIR/delay/CFO and combined
        nonlinear/IQ/ADC effects, compare amplitude and power noise controls,
        verify active-burst SNR for SISO/MIMO, require repeatable random state,
        validate hidden PA power calibration, and reject invalid settings.

    Returns:
        result: None. Assertions enforce the documented channel contract.
    """

    testSignal = np.array(
        [0.2 + 0.3j, -0.4 + 0.1j, 0.05 - 0.2j],
        dtype=np.complex128,
    )
    for phaseDegrees, phaseFactor in (
        (-90, -1j),
        (0, 1.0 + 0.0j),
        (90, 1j),
    ):
        phaseChannel = Channel(
            parameters={
                "phaseDegrees": phaseDegrees,
                "width": 0,
            }
        )
        phaseOutput = phaseChannel.ProcessPaOutput(testSignal)
        assert np.allclose(phaseOutput, phaseFactor * testSignal)

    # Forward instrument sampling must ignore every embedded feedback-only
    # defect, while ideal feedback mode must remain exactly transparent.
    forwardChannel = Channel(
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": 20.0e6,
            "fbGainDb": 8.0,
            "fbPhaseDegrees": 37.0,
            "fbFirTaps": (1.0 + 0.0j, 0.3 - 0.2j),
            "fbIntegerDelaySamples": 2,
            "fbFractionalDelaySamples": 0.2,
            "fbCarrierFrequencyOffsetHz": 1500.0,
            "fbSamplingFrequencyOffsetPpm": 25.0,
            "fbIqGainImbalanceDb": 1.0,
            "fbIqPhaseImbalanceDegrees": 4.0,
            "fbDcOffset": 0.02 - 0.01j,
            "fbThirdOrderCoefficient": -0.1 + 0.02j,
            "fbClipAmplitude": 0.5,
            "fbAdcWidth": 8,
            "fbAdcFullScale": 0.6,
            "width": 0,
        }
    )
    assert forwardChannel.sampleMode == "forward"
    assert np.array_equal(
        forwardChannel.ProcessPaOutput(testSignal),
        testSignal,
    )
    idealFeedbackChannel = Channel(
        parameters={"sampleMode": "FB", "width": 0}
    )
    assert idealFeedbackChannel.sampleMode == "fb"
    assert np.array_equal(
        idealFeedbackChannel.ProcessPaOutput(testSignal),
        testSignal,
    )

    linearFeedbackInput = np.asarray(
        (1.0 + 0.0j, 0.4 - 0.2j, -0.1 + 0.3j, 0.0 + 0.0j),
        dtype=np.complex128,
    )
    feedbackFirTaps = np.asarray(
        (1.0 + 0.0j, 0.5j), dtype=np.complex128
    )
    linearFeedbackChannel = Channel(
        parameters={
            "sampleMode": "fb",
            "fbGainDb": 20.0 * np.log10(2.0),
            "fbPhaseDegrees": 90.0,
            "fbFirTaps": tuple(feedbackFirTaps),
            "fbIntegerDelaySamples": 1,
            "width": 0,
        }
    )
    expectedFilteredSignal = np.convolve(
        linearFeedbackInput,
        feedbackFirTaps,
        mode="full",
    )[: linearFeedbackInput.size]
    expectedLinearFeedback = np.r_[
        0.0 + 0.0j,
        (2.0j * expectedFilteredSignal)[:-1],
    ]
    assert np.allclose(
        linearFeedbackChannel.ProcessPaOutput(linearFeedbackInput),
        expectedLinearFeedback,
    )

    carrierSampleRateHz = 10.0e6
    carrierOffsetHz = 125.0e3
    carrierInput = np.ones(128, dtype=np.complex128)
    carrierChannel = Channel(
        parameters={
            "sampleMode": "fb",
            "sampleRateHz": carrierSampleRateHz,
            "fbCarrierFrequencyOffsetHz": carrierOffsetHz,
            "width": 0,
        }
    )
    expectedCarrierPhasor = np.exp(
        1j
        * 2.0
        * np.pi
        * carrierOffsetHz
        * np.arange(carrierInput.size)
        / carrierSampleRateHz
    )
    assert np.allclose(
        carrierChannel.ProcessPaOutput(carrierInput),
        expectedCarrierPhasor,
    )

    combinedFeedbackChannel = Channel(
        parameters={
            "sampleMode": "fb",
            "sampleRateHz": 40.0e6,
            "fbFractionalDelaySamples": 0.25,
            "fbSamplingFrequencyOffsetPpm": 40.0,
            "fbIqGainImbalanceDb": 1.5,
            "fbIqPhaseImbalanceDegrees": 5.0,
            "fbDcOffset": 0.02 - 0.01j,
            "fbThirdOrderCoefficient": -0.15 + 0.03j,
            "fbClipAmplitude": 0.5,
            "fbAdcWidth": 8,
            "fbAdcFullScale": 0.6,
            "width": 0,
        }
    )
    combinedFeedbackInput = np.tile(testSignal, 128)
    combinedFeedbackOutput = (
        combinedFeedbackChannel.ProcessPaOutput(
            combinedFeedbackInput
        )
    )
    adcStep = 0.6 / float(2 ** 7)
    assert combinedFeedbackOutput.shape == combinedFeedbackInput.shape
    assert not np.allclose(
        combinedFeedbackOutput, combinedFeedbackInput
    )
    assert np.max(combinedFeedbackOutput.real) <= 0.6
    assert np.max(combinedFeedbackOutput.imag) <= 0.6
    assert np.allclose(
        combinedFeedbackOutput.real / adcStep,
        np.rint(combinedFeedbackOutput.real / adcStep),
    )
    assert np.allclose(
        combinedFeedbackOutput.imag / adcStep,
        np.rint(combinedFeedbackOutput.imag / adcStep),
    )

    # Post-PA coupling must preserve both direct paths and add the delayed
    # complex leakage only to its configured destination column.
    couplingSampleCount = 64
    couplingSource0 = np.exp(
        1j
        * 2.0
        * np.pi
        * np.arange(couplingSampleCount, dtype=float)
        / 13.0
    )
    couplingSource1 = 0.4 * np.exp(
        1j
        * 2.0
        * np.pi
        * np.arange(couplingSampleCount, dtype=float)
        / 19.0
    )
    uncoupledPaMatrix = np.column_stack(
        (couplingSource0, couplingSource1)
    )
    postCouplingTaps = np.asarray(
        (1.0 + 0.0j, 0.25 - 0.10j),
        dtype=np.complex128,
    )
    postCouplingChannel = Channel(
        parameters={
            "postPaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": 20.0 * np.log10(0.2),
                    "phaseDegrees": 90.0,
                    "integerDelaySamples": 2,
                    "fractionalDelaySamples": 0.0,
                    "firTaps": tuple(postCouplingTaps),
                },
            ),
            "width": 0,
        }
    )
    filteredCoupling = np.convolve(
        couplingSource0,
        postCouplingTaps,
        mode="full",
    )[:couplingSampleCount]
    expectedLeakage = np.r_[
        np.zeros(2, dtype=np.complex128),
        (0.2j * filteredCoupling)[:-2],
    ]
    expectedPostCoupling = uncoupledPaMatrix.copy()
    expectedPostCoupling[:, 1] += expectedLeakage
    actualPostCoupling = postCouplingChannel.ProcessPaOutput(
        uncoupledPaMatrix
    )
    assert np.allclose(actualPostCoupling, expectedPostCoupling)
    assert np.array_equal(
        actualPostCoupling[:, 0], uncoupledPaMatrix[:, 0]
    )

    # Different pre-PA directions may use independent delays and amplitudes.
    asymmetricPreChannel = Channel(
        parameters={
            "prePaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": -20.0,
                    "integerDelaySamples": 1,
                },
                {
                    "sourceChain": 1,
                    "destinationChain": 0,
                    "gainDb": -26.0,
                    "phaseDegrees": -35.0,
                    "integerDelaySamples": 3,
                    "fractionalDelaySamples": 0.20,
                },
            ),
            "width": 0,
        }
    )
    asymmetricPreOutput = asymmetricPreChannel.ApplyPrePaCoupling(
        uncoupledPaMatrix
    )
    assert asymmetricPreOutput.shape == uncoupledPaMatrix.shape
    assert not np.allclose(
        asymmetricPreOutput[:, 0], uncoupledPaMatrix[:, 0]
    )
    assert not np.allclose(
        asymmetricPreOutput[:, 1], uncoupledPaMatrix[:, 1]
    )
    assert asymmetricPreChannel.HasPrePaCoupling()

    noiseSampleCount = 200000
    zeroSignal = np.zeros(noiseSampleCount, dtype=np.complex128)
    amplitudeChannel = Channel(
        parameters={
            "noiseAmpMv": 10.0,
            "maximumOutputPowerDbm": 25.0,
            "loadResistanceOhm": 50.0,
            "randomSeed": 73,
            "width": 0,
        }
    )
    amplitudeNoise = amplitudeChannel.ProcessPaOutput(zeroSignal)
    fullScaleRmsVolts = (
        np.sqrt(1.0e-3 * 50.0) * 10.0 ** (25.0 / 20.0)
    )
    measuredNoiseMv = (
        np.sqrt(np.mean(np.abs(amplitudeNoise) ** 2))
        * fullScaleRmsVolts
        * 1.0e3
    )
    assert abs(measuredNoiseMv - 10.0) <= 0.10
    amplitudeChannel.ResetRandomGenerator()
    repeatedNoise = amplitudeChannel.ProcessPaOutput(zeroSignal)
    assert np.array_equal(amplitudeNoise, repeatedNoise)

    equivalentNoisePowerDbm = float(
        20.0 * np.log10(10.0e-3)
        - 10.0 * np.log10(50.0e-3)
    )
    powerChannel = Channel(
        parameters={
            "noisePwrDbm": equivalentNoisePowerDbm,
            "maximumOutputPowerDbm": 25.0,
            "loadResistanceOhm": 50.0,
            "randomSeed": 73,
            "width": 0,
        }
    )
    assert np.isclose(
        amplitudeChannel.ResolveNoiseRmsVolts(),
        powerChannel.ResolveNoiseRmsVolts(),
    )
    assert np.isclose(
        amplitudeChannel.ResolveNoiseRmsNormalized(),
        powerChannel.ResolveNoiseRmsNormalized(),
    )

    activeSampleCount = 120000
    activeSamples = np.exp(
        1j
        * 2.0
        * np.pi
        * np.arange(activeSampleCount, dtype=float)
        / 37.0
    )
    paddedSignal = np.r_[
        np.zeros(1000, dtype=np.complex128),
        activeSamples,
        np.zeros(2000, dtype=np.complex128),
    ]
    snrChannel = Channel(
        parameters={
            "noiseSnrDb": 30.0,
            "randomSeed": 83,
            "width": 0,
        }
    )
    snrOutput = snrChannel.ProcessPaOutput(paddedSignal)
    activeSlice = slice(1000, 1000 + activeSampleCount)
    activeNoise = (
        snrOutput[activeSlice] - paddedSignal[activeSlice]
    )
    measuredSnrDb = float(
        10.0
        * np.log10(
            np.mean(np.abs(paddedSignal[activeSlice]) ** 2)
            / np.mean(np.abs(activeNoise) ** 2)
        )
    )
    assert abs(measuredSnrDb - 30.0) <= 0.10

    mimoPaddedSignal = np.column_stack(
        (paddedSignal, 0.25 * paddedSignal)
    )
    mimoSnrChannel = Channel(
        parameters={
            "noiseSnrDb": 24.0,
            "randomSeed": 89,
            "width": 0,
        }
    )
    mimoSnrOutput = mimoSnrChannel.ProcessPaOutput(
        mimoPaddedSignal
    )
    mimoActiveSignal = mimoPaddedSignal[activeSlice, :]
    mimoActiveNoise = (
        mimoSnrOutput[activeSlice, :] - mimoActiveSignal
    )
    measuredMimoSnrDb = 10.0 * np.log10(
        np.mean(np.abs(mimoActiveSignal) ** 2, axis=0)
        / np.mean(np.abs(mimoActiveNoise) ** 2, axis=0)
    )
    assert np.all(np.abs(measuredMimoSnrDb - 24.0) <= 0.10)

    paModel = PaModel(parameters={"modelName": "wiener", "width": 0})
    paChannel = Channel(
        paModel=paModel,
        parameters={
            "phaseDegrees": 90,
            "width": 0,
        },
    )
    expectedOutput = 1j * paModel.Process(testSignal)
    assert np.allclose(paChannel.Process(testSignal), expectedOutput)
    assert np.allclose(
        paChannel.SmallSignalGain(),
        1j * paModel.SmallSignalGain(),
    )
    feedbackPaChannel = Channel(
        paModel=paModel,
        parameters={
            "sampleMode": "fb",
            "fbGainDb": 20.0 * np.log10(2.0),
            "fbPhaseDegrees": 90.0,
            "width": 0,
        },
    )
    assert np.allclose(
        feedbackPaChannel.SmallSignalGain(),
        2.0j * paModel.SmallSignalGain(),
    )

    # A user-facing calibrated call accepts an arbitrarily scaled burst and
    # target output power. Channel must hide all repeated PA evaluations,
    # exclude padding from the detector, and add impairments only after the
    # clean PA output has met the target.
    burstSamples = 1.7 * np.exp(
        1j * 2.0 * np.pi * np.arange(1024, dtype=float) / 29.0
    )
    rawBurst = np.r_[
        np.zeros(80, dtype=np.complex128),
        burstSamples,
        np.zeros(120, dtype=np.complex128),
    ]
    calibratedChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.10,
            "width": 0,
        },
    )
    calibratedOutput = calibratedChannel.Process(
        rawBurst,
        outputPowerDbm=18.0,
    )
    calibrationMetrics = (
        calibratedChannel.GetLastCalibrationMetrics()
    )
    assert calibrationMetrics["converged"]
    assert abs(
        calibrationMetrics["measuredOutputPowerDbmPerChain"][0]
        - 18.0
    ) <= 0.10
    assert np.array_equal(
        calibratedOutput,
        calibratedChannel.GetLastPaOutput(),
    )
    assert calibratedChannel.GetLastPaInput().shape == rawBurst.shape

    # A target sequence jointly calibrates different PA families while weak
    # pre-PA coupling makes every drive affect both measured output powers.
    mimoRawBurst = np.column_stack((rawBurst, 0.35 * rawBurst))
    mimoChannel = Channel(
        paModel=MimoPaModel(
            parameters={
                "numTransmitChains": 2,
                "paParametersPerChain": (
                    {"modelName": "doherty"},
                    {"modelName": "gmp"},
                ),
                "width": 0,
            }
        ),
        parameters={
            "prePaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": -20.0,
                    "phaseDegrees": 15.0,
                    "integerDelaySamples": 2,
                },
                {
                    "sourceChain": 1,
                    "destinationChain": 0,
                    "gainDb": -24.0,
                    "phaseDegrees": -20.0,
                    "integerDelaySamples": 1,
                },
            ),
            "postPaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": -22.0,
                    "integerDelaySamples": 1,
                },
            ),
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.15,
            "width": 0,
        },
    )
    mimoOutput = mimoChannel.Process(
        mimoRawBurst,
        outputPowerDbm=(17.0, 19.0),
    )
    mimoCalibrationMetrics = (
        mimoChannel.GetLastCalibrationMetrics()
    )
    assert mimoOutput.shape == mimoRawBurst.shape
    assert (
        mimoChannel.GetParameters()["jointPowerCalibration"]
        is None
    )
    assert (
        mimoChannel._powerCalibration.GetParameters()[
            "enableJointCalibration"
        ]
        is True
    )
    assert np.all(
        np.abs(
            np.asarray(
                mimoCalibrationMetrics[
                    "measuredOutputPowerDbmPerChain"
                ]
            )
            - np.asarray((17.0, 19.0))
        )
        <= 0.15
    )

    fixedPa = PaModel(parameters={"modelName": "gmp", "width": 16})
    fixedChannel = Channel(
        paModel=fixedPa,
        parameters={
            "phaseDegrees": -90,
            "noiseAmpMv": 10.0,
            "randomSeed": 91,
            "width": 16,
        },
    )
    fixedInput = FixedPoint(16).EncodeComplex(testSignal)
    fixedOutput = fixedChannel.Process(fixedInput)
    assert fixedOutput.dtype == np.complex128
    assert np.array_equal(fixedOutput.real, np.rint(fixedOutput.real))
    assert np.array_equal(fixedOutput.imag, np.rint(fixedOutput.imag))

    invalidConfigurations = (
        {"sampleMode": "instrument"},
        {"sampleRateHz": 0.0},
        {"phaseDegrees": 45},
        {"noiseAmpMv": -1.0},
        {"noiseAmpMv": 10.0, "noisePwrDbm": -27.0},
        {"noiseAmpMv": 10.0, "noiseSnrDb": 30.0},
        {"noisePwrDbm": -27.0, "noiseSnrDb": 30.0},
        {
            "noiseAmpMv": 10.0,
            "noisePwrDbm": -27.0,
            "noiseSnrDb": 30.0,
        },
        {"noiseSnrDb": np.nan},
        {"fbIntegerDelaySamples": -1},
        {"fbFractionalDelaySamples": 0.5},
        {"fbSamplingFrequencyOffsetPpm": 1.0e6},
        {"fbFirTaps": ()},
        {"fbThirdOrderCoefficient": complex(np.nan, 0.0)},
        {"fbClipAmplitude": 0.0},
        {"fbAdcWidth": 1},
        {"fbAdcFullScale": 0.0},
        {"prePaCouplingPaths": "invalid"},
        {
            "prePaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 0,
                },
            )
        },
        {
            "postPaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "fractionalDelaySamples": 0.5,
                },
            )
        },
        {"jointPowerCalibration": "auto"},
        {"calibrationProbeStepDb": 0.0},
        {"calibrationRegularization": 0.0},
    )
    for invalidParameters in invalidConfigurations:
        try:
            Channel(parameters=invalidParameters)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f"invalid channel configuration accepted: "
                f"{invalidParameters!r}"
            )


def CheckTwoToneIlcAnalysis() -> None:
    """Verify two-tone generation, IM analysis, ILC, and all-method reporting.

    Processing details:
        Algorithm: Check exact IM3/IM5/IM7 frequency construction, floating and
        fixed public interfaces, a nearly product-free ideal waveform, visible
        Wiener-PA intermodulation, equal-power complex-gain ILC suppression,
        dictionary result keys, and a small complete benchmark with serialized
        and graphical artifacts.

    Returns:
        result: None. Assertion failures identify two-tone regressions.
    """

    with warnings.catch_warnings(record=True) as capturedWarnings:
        warnings.simplefilter("always")
        warningGenerator = WaveGenTwoTone(
            parameters={
                "width": 0,
                "unsupportedToneOption": 1,
            }
        )
    assert warningGenerator.width == 0
    assert any(
        "unsupportedToneOption" in str(warningRecord.message)
        for warningRecord in capturedWarnings
    )

    floatingWaveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": 100.0e6,
            "toneFrequenciesHz": (-2.0e6, 2.0e6),
            "numSamples": 4096,
            "width": 0,
        }
    ).Generate()
    assert floatingWaveform.IntermodulationFrequencies(3) == (
        -6.0e6,
        6.0e6,
    )
    assert floatingWaveform.IntermodulationFrequencies(5) == (
        -10.0e6,
        10.0e6,
    )
    assert floatingWaveform.IntermodulationFrequencies(7) == (
        -14.0e6,
        14.0e6,
    )
    fixedWaveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": 100.0e6,
            "toneFrequenciesHz": (-2.0e6, 2.0e6),
            "numSamples": 4096,
            "width": 16,
        }
    ).Generate()
    assert np.all(
        fixedWaveform.samples.real
        == np.rint(fixedWaveform.samples.real)
    )
    assert np.all(
        fixedWaveform.samples.imag
        == np.rint(fixedWaveform.samples.imag)
    )
    resultAnalysis = TwoToneAnalysis(
        floatingWaveform,
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    idealMetrics = resultAnalysis.Analyze(floatingWaveform.samples)
    assert isinstance(idealMetrics, dict)
    assert idealMetrics["im3WorstDbc"] < -100.0
    assert idealMetrics["im5WorstDbc"] < idealMetrics["im3WorstDbc"]
    assert idealMetrics["im7WorstDbc"] < idealMetrics["im5WorstDbc"]

    paModel = PaModel(parameters={"modelName": "wiener", "width": 0})
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "outputPowerDbm": 20.0,
            "maximumOutputPowerDbm": 25.0,
            "width": 0,
        },
    )
    referenceSignal = powerCalibration.Calibrate(
        floatingWaveform.samples
    )
    baselineOutput = powerCalibration.GetLastPaOutput()
    baselineMetrics = resultAnalysis.Analyze(baselineOutput)
    assert -40.0 < baselineMetrics["im3WorstDbc"] < -20.0
    ilcResult = RunComplexGainIlc(
        referenceSignal,
        paModel,
        ILCConfig(
            numIterations=5,
            learningRate=0.15,
            maxAmplitude=1.5,
            randomSeed=313,
        ),
        floatingWaveform.sampleRateHz,
    )
    analyzedIlc = resultAnalysis.AnalyzeIlcHistory(ilcResult.history)
    powerCalibration.Calibrate(analyzedIlc.bestInputSignal)
    selectedMetrics = resultAnalysis.Analyze(
        powerCalibration.GetLastPaOutput()
    )
    assert (
        selectedMetrics["im3WorstDbc"]
        < baselineMetrics["im3WorstDbc"] - 2.0
    )
    assert (
        selectedMetrics["im5WorstDbc"]
        < baselineMetrics["im5WorstDbc"] - 2.0
    )
    assert abs(selectedMetrics["outputPowerDbm"] - 20.0) <= 0.25

    from tests.BenchMark import (
        RunTwoToneIlcBenchmark,
        TwoToneBenchmarkConfig,
    )

    with TemporaryDirectory() as temporaryDirectory:
        outputDirectory = Path(temporaryDirectory)
        benchmarkRows = RunTwoToneIlcBenchmark(
            TwoToneBenchmarkConfig(
                width=0,
                numSamples=4096,
                numIterations=2,
                outputDirectory=outputDirectory,
            )
        )
        assert len(benchmarkRows) == 8
        assert (
            outputDirectory / "all_ilc_two_tone_metrics.csv"
        ).exists()
        assert (
            outputDirectory / "all_ilc_two_tone_metrics.json"
        ).exists()
        assert (
            outputDirectory / "all_ilc_two_tone_imd.png"
        ).exists()


def CheckPaCharacterizationBenchmark() -> None:
    """Verify multi-model two-tone PA feature sweeps and all artifacts.

    Processing details:
        Algorithm: Run a compact Wiener/GMP/Doherty frequency and tone-spacing
        sweep, require the expected point counts and finite summary metrics,
        verify equal-power nonlinear measurements, require five complete
        measurement-backed DPD recommendations per model, and check all CSV,
        JSON, and PNG outputs plus the dedicated principle document.

    Returns:
        result: None. Assertion failures identify PA characterization
            regressions.
    """

    from tests.BenchMark import (
        PaCharacterizationConfig,
        RunPaCharacterizationBenchmark,
    )

    with TemporaryDirectory() as temporaryDirectory:
        outputDirectory = Path(temporaryDirectory)
        result = RunPaCharacterizationBenchmark(
            PaCharacterizationConfig(
                sampleRateHz=100.0e6,
                frequencyCentersHz=(-10.0e6, 0.0, 10.0e6),
                frequencyToneSpacingHz=2.0e6,
                memoryToneSpacingsHz=(0.5e6, 2.0e6, 4.0e6),
                dynamicToneSpacingHz=2.0e6,
                powerSweepDbm=(15.0, 20.0),
                numSamples=4096,
                settlingSamples=64,
                outputPowerDbm=20.0,
                width=0,
                outputDirectory=outputDirectory,
            )
        )
        assert len(result.frequencyResponse) == 18
        assert len(result.memoryEffect) == 9
        assert len(result.powerSweep) == 6
        assert len(result.summaries) == 3
        assert len(result.recommendations) == 15
        resultDocument = result.ToDict()
        assert len(resultDocument["powerSweep"]) == 6
        assert len(resultDocument["recommendations"]) == 15
        assert tuple(
            summary.modelName for summary in result.summaries
        ) == ("wiener", "gmp", "doherty")
        for summary in result.summaries:
            summaryValues = tuple(
                value
                for key, value in summary.ToDict().items()
                if key != "modelName"
            )
            assert np.all(np.isfinite(summaryValues))
            assert summary.gainRippleDb >= 0.0
            assert summary.phaseNonlinearityDegrees >= 0.0
            assert summary.im3SpacingVariationDb >= 0.0
            assert summary.maximumIm3AsymmetryDb >= 0.0
        for memoryPoint in result.memoryEffect:
            assert (
                abs(memoryPoint.outputPowerDbm - 20.0) <= 0.25
            )
        for powerPoint in result.powerSweep:
            assert (
                abs(
                    powerPoint.measuredOutputPowerDbm
                    - powerPoint.targetOutputPowerDbm
                )
                <= 0.25
            )
        for modelName in ("wiener", "gmp", "doherty"):
            assert tuple(
                powerPoint.targetOutputPowerDbm
                for powerPoint in result.powerSweep
                if powerPoint.modelName == modelName
            ) == (15.0, 20.0)
            modelRecommendations = tuple(
                recommendation
                for recommendation in result.recommendations
                if recommendation.modelName == modelName
            )
            assert tuple(
                recommendation.testName
                for recommendation in modelRecommendations
            ) == (
                "frequency_response",
                "memory_effect",
                "dynamic_hysteresis",
                "nominal_nonlinearity",
                "output_power",
            )
            for recommendation in modelRecommendations:
                recommendationRow = recommendation.ToDict()
                for fieldName in (
                    "measuredEvidence",
                    "dpdArchitecture",
                    "dpdConfiguration",
                    "trainingStrategy",
                    "acceptanceCriteria",
                ):
                    assert recommendationRow[fieldName]
        for artifactName in (
            "pa_frequency_response.csv",
            "pa_memory_effect.csv",
            "pa_power_sweep.csv",
            "pa_characterization_summary.csv",
            "pa_dpd_recommendations.csv",
            "pa_characterization.json",
            "pa_frequency_response.png",
            "pa_memory_effect.png",
            "pa_nonlinearity_comparison.png",
            "pa_power_characteristics.png",
        ):
            assert (outputDirectory / artifactName).exists()
        recommendationDocument = json.loads(
            (
                outputDirectory / "pa_characterization.json"
            ).read_text(encoding="utf-8")
        )
        assert len(recommendationDocument["recommendations"]) == 15
        dpdGmpDirectory = outputDirectory / "dpd_gmp"
        for artifactName in (
            "dpd_gmp_stage_metrics.csv",
            "dpd_gmp_improvement_comparison.csv",
            "dpd_gmp_benchmark.json",
            "dpd_gmp_performance.png",
        ):
            assert (dpdGmpDirectory / artifactName).exists()
        dpdGmpDocument = json.loads(
            (
                dpdGmpDirectory / "dpd_gmp_benchmark.json"
            ).read_text(encoding="utf-8")
        )
        assert len(dpdGmpDocument["stages"]) == 8
        assert all(
            comparison["expectationMet"]
            for comparison in dpdGmpDocument["comparisons"]
        )
    paAnalysisDocument = (
        GetProjectRoot() / "doc" / "PaAnalyse.md"
    ).read_text(encoding="utf-8")
    for requiredText in (
        "小信号频率响应",
        "双音间隔扫描",
        "动态AM-AM/AM-PM迟滞",
        "输出功率扫描",
        "小信号频响测试后的DPD建议",
        "双音间隔测试后的DPD建议",
        "动态迟滞测试后的DPD建议",
        "标称非线性测试后的DPD建议",
        "输出功率测试后的DPD建议",
        "pa_dpd_recommendations.csv",
        "测试结果",
        "pa_frequency_response.png",
        "pa_memory_effect.png",
        "pa_nonlinearity_comparison.png",
        "pa_power_characteristics.png",
    ):
        assert requiredText in paAnalysisDocument


def CheckDpdGmpModelAndBenchmark() -> None:
    """Verify DpdGmp training modes and every staged performance artifact.

    Processing details:
        Algorithm: Check floating identity inference, direct nonlinear fitting,
        retained multi-segment weights, fixed-point public codes, unknown-key
        warnings, and the deterministic PA-analysis-driven benchmark whose
        target metric must improve for every declared method change.

    Returns:
        result: None. Assertions identify GMP model or benchmark regressions.
    """

    randomGenerator = np.random.default_rng(701)
    referenceSignal = (
        randomGenerator.standard_normal(2048)
        + 1j * randomGenerator.standard_normal(2048)
    )
    referenceSignal *= 0.18 / np.sqrt(
        np.mean(np.abs(referenceSignal) ** 2)
    )
    identityDpd = DpdGmp(
        parameters={
            "width": 0,
            "maximumOutputMagnitude": None,
        }
    )
    assert np.array_equal(
        identityDpd.Process(referenceSignal),
        referenceSignal,
    )
    featureCount = len(identityDpd.GetFeatureSpecs())
    assert featureCount == 4 * 3 + 2 * 3 * 3 * 2

    targetSignal = (
        1.08 * referenceSignal
        + 0.16 * referenceSignal * np.abs(referenceSignal) ** 2
    )
    fittedDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "ridgeFactor": 1.0e-8,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    beforeNmseDb = fittedDpd.CalculateNmse(
        referenceSignal,
        targetSignal,
    )
    trainingResult = fittedDpd.Fit(
        referenceSignal,
        targetSignal,
    )
    afterNmseDb = fittedDpd.CalculateNmse(
        referenceSignal,
        targetSignal,
    )
    assert afterNmseDb < beforeNmseDb - 40.0
    assert trainingResult.afterNmseDb < trainingResult.beforeNmseDb
    assert fittedDpd.GetLastTrainingResult() == trainingResult
    assert trainingResult.featureCount == 2

    firstOrderParameters = {
        "nonlinearOrders": (1,),
        "memoryDepth": 1,
        "crossMemoryDepth": 0,
        "ridgeFactor": 1.0e-8,
        "maximumOutputMagnitude": None,
        "width": 0,
    }
    uniformDpd = DpdGmp(parameters=firstOrderParameters)
    uniformDpd.FitSegments(
        (referenceSignal, referenceSignal),
        (referenceSignal, 2.0 * referenceSignal),
        segmentWeights=(1.0, 1.0),
    )
    weightedDpd = DpdGmp(parameters=firstOrderParameters)
    weightedDpd.FitSegments(
        (referenceSignal, referenceSignal),
        (referenceSignal, 2.0 * referenceSignal),
        segmentWeights=(1.0, 4.0),
    )
    assert (
        weightedDpd.GetCoefficients()[0].real
        > uniformDpd.GetCoefficients()[0].real + 0.20
    )

    fixedWaveform = WaveGenWifi(
        parameters={
            "frameFormat": "EHT",
            "bandwidthMhz": 20,
            "numDataSymbols": 1,
            "width": 16,
        }
    ).Generate()
    fixedDpd = DpdGmp(
        parameters={
            "width": 16,
        }
    )
    fixedOutput = fixedDpd.Process(fixedWaveform.samples)
    assert fixedOutput.dtype == np.complex128
    assert np.max(np.abs(fixedOutput.real)) > 1.0
    assert np.array_equal(fixedOutput, fixedWaveform.samples)
    with warnings.catch_warnings(record=True) as warningRecords:
        warnings.simplefilter("always")
        warnedDpd = DpdGmp(
            parameters={
                "width": 0,
                "unknownGmpOption": 9,
            }
        )
    assert warnedDpd.width == 0
    assert any(
        "unknownGmpOption" in str(warningRecord.message)
        for warningRecord in warningRecords
    )

    for documentName, requiredText in (
        ("DPD-GMP.md", "加权岭回归"),
        ("DpdGmp.md", "多功率联合训练"),
        ("PaAnalyse.md", "PA特性分析后的DPD-GMP改进与实测对比"),
    ):
        documentText = (
            GetProjectRoot() / "doc" / documentName
        ).read_text(encoding="utf-8")
        assert requiredText in documentText


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
    CheckDocumentationApiConsistency()
    CheckInternalDefaultConfiguration()
    CheckUnknownConfigurationWarnings()
    CheckFixedPointInterfaces()
    CheckChannelModel()
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
    CheckDohertyPaModel()
    CheckIlcImprovement()
    CheckIlcFeedbackSynchronization()
    CheckReceiveOnlyWifiAnalysis()
    CheckMseEvmConvergence()
    CheckTwoToneIlcAnalysis()
    CheckPaCharacterizationBenchmark()
    CheckDpdGmpModelAndBenchmark()
    print("All DPD-ILC project checks passed.")


if __name__ == "__main__":
    RunTests()
