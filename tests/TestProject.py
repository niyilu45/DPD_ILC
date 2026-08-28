"""Self-contained project checks that preserve the requested naming style."""

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Sequence
from unittest.mock import patch
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

from inc.lib.Analysis import Analysis, AveragePeriodogram
from inc.lib.Channel import Channel
from inc.lib.ChannelAnalyse import ChannelAnalyse
from inc.lib.DpdGmp import (
    AugmentedDpdGmp,
    CouplingAwareDpdGmp,
    DpdGmp,
    PiecewiseDpdGmp,
)
from inc.lib.DpdLms import DpdLms
from inc.lib.DpdIlc import (
    CalculateIterationMetrics,
    FitMimoGmpPredistorter,
    ILCConfig,
    ILCResult,
    MimoIlcResult,
    NormalizedPaAdapter,
    RunAugmentedIqIlc,
    RunComplexGainIlc,
    RunDirectionalGaussNewtonIlc,
    RunFirIlc,
    RunFrequencyDomainIlc,
    RunMimoFrequencyDomainIlc,
    RunParameterDomainIlc,
    RunScalarPIlc,
)
from inc.lib.Fec import (
    BuildDescriptorLdpcMatrices,
    DecodeDescriptorLdpc,
    EncodeDescriptorLdpc,
)
from inc.utils.Draw import Draw
from inc.utils.FixedPoint import (
    FixedPoint,
    FixedPointArray,
    GetFixedPointFormat,
)
from inc.lib.PaModel import (
    DefaultGmpCoefficients,
    DohertyConfig,
    DohertyPA,
    DelaySignal,
    GMPConfig,
    GMPPA,
    IQImbalancePA,
    MimoPaModel,
    PaModel,
    PiecewiseGMPConfig,
    PiecewiseGMPPA,
    RappConfig,
    RappPA,
    ThermalConfig,
    ThermalNetwork,
    WienerConfig,
)
from inc.lib.ParseWifi import (
    BuildWifiDescriptorBits,
    DecodeWifiDescriptorBits,
    DescriptorLdpcPhysicalLayout,
    ParseWifi,
)
from inc.utils.SigProc import (
    FeedbackIqCalibration,
    PowerCalibration,
    SigProc,
)
from inc.lib.WaveGenWifi import (
    NormalizeFrameFormat,
    WaveGenWifi,
)
from inc.lib.WaveGenTwoTone import WaveGenTwoTone
from inc.lib.TwoToneAnalysis import TwoToneAnalysis
from main import EvaluateIlcPowerPoint


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
    dpdLmsSource = (
        projectRoot / "inc" / "lib" / "DpdLms.py"
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
        "DpdLms.py",
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
    assert "class DpdLms(DpdGmp):" in dpdLmsSource
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
        "from lib.DpdLms import DpdLms; "
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
        "DpdLmsBenchmarkConfig",
        "DpdLmsBenchmarkResult",
        "BuildDpdLmsTarget",
        "RunDpdLmsBenchmark",
        "SaveDpdLmsBenchmarkResult",
        "PrintDpdLmsBenchmarkResult",
        "ChannelAnalysisBenchmarkConfig",
        "ChannelDpdStageResult",
        "ChannelDpdImprovement",
        "ChannelAnalysisBenchmarkResult",
        "RunChannelAnalysisBenchmark",
        "SaveChannelAnalysisResults",
        "PrintChannelAnalysisResults",
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
        "H类：Rapp/Wiener/GMP/Doherty PA双音特性",
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
    assert (
        "J类：通道测量与耦合感知 DPD-GMP"
        in benchmarkDocument
    )
    assert (
        "L类：DPD-LMS逐样点更新与漂移跟踪"
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

    documentPaths = [GetProjectRoot() / "README.md"]
    documentPaths.extend(
        sorted((GetProjectRoot() / "doc").glob("*.md"))
    )
    forbiddenMacros = (
        r"\operatorname",
        r"\text",
        r"\dfrac",
        r"\tfrac",
        r"\mathop",
        r"\begin{aligned}",
        r"\begin{align",
        r"\begin{array}",
        r"\begin{cases}",
        r"\substack",
        r"\overset",
        r"\underset",
        r"\underbrace",
        r"\overbrace",
        r"\left.",
        r"\right.",
        r"\tag",
        r"\label",
        r"\eqref",
        r"\newcommand",
        r"\require",
        r"\unicode",
    )
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


def CheckDocumentationImageLinks() -> None:
    """Verify that every repository-local Markdown image can render on GitHub.

    Processing details:
        Algorithm: Scan README and principle documents for Markdown image
        targets, ignore explicit web URLs, strip optional anchors, resolve each
        path relative to its source document, and require a nonempty file.

    Returns:
        result: None. Missing or empty image assets fail before publication.
    """

    documentPaths = [GetProjectRoot() / "README.md"]
    documentPaths.extend(
        sorted((GetProjectRoot() / "doc").glob("*.md"))
    )
    imagePattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    checkedImageCount = 0
    for documentPath in documentPaths:
        markdownText = documentPath.read_text(encoding="utf-8")
        for rawTarget in imagePattern.findall(markdownText):
            imageTarget = rawTarget.strip().split("#", 1)[0]
            if imageTarget.startswith(("http://", "https://", "data:")):
                continue
            imagePath = (documentPath.parent / imageTarget).resolve()
            assert imagePath.is_file(), (
                f"missing Markdown image in {documentPath}: {imageTarget}"
            )
            assert imagePath.stat().st_size > 0, (
                f"empty Markdown image in {documentPath}: {imageTarget}"
            )
            checkedImageCount += 1
    assert checkedImageCount >= 19


def CheckDocumentationApiConsistency() -> None:
    """Verify runnable Markdown snippets and documented Analysis arguments.

    Processing details:
        Algorithm: Compile every fenced Python example, compare the public
        ``Analysis`` constructor parameter order with its documented
        signature, require synchronization examples to use the explicit
        ``signalProcessingParameters`` argument, require the piecewise-GMP PA
        example to use the common ``parameters`` mapping, and retain one
        documented compatibility note for the legacy nested mapping form.

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
        "outputFullScaleAmplitude",
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
        "channelBandwidthHz=None, width=None, "
        "outputFullScaleAmplitude=None, **parameterOverrides)"
    )
    readmeText = (projectRoot / "README.md").read_text(encoding="utf-8")
    analysisDocumentText = (
        projectRoot / "doc" / "Analysis.md"
    ).read_text(encoding="utf-8")
    signalDocumentText = (
        projectRoot / "doc" / "SigProc.md"
    ).read_text(encoding="utf-8")
    paModelDocumentText = (
        projectRoot / "doc" / "PaModel.md"
    ).read_text(encoding="utf-8")
    assert expectedSignatureText in readmeText
    assert (
        'piecewisePa = PaModel(\n'
        '    parameters={\n'
        '        "modelName": "piecewise_gmp",'
    ) in paModelDocumentText
    assert (
        'piecewisePa = PaModel(\n'
        '    modelName="piecewise_gmp",'
    ) not in paModelDocumentText
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
                "outputFullScaleAmplitude",
                "parameterOverrides",
            ),
            (
                "TwoToneAnalysis(waveform, parameters=None, width=None, "
                "outputFullScaleAmplitude=None, "
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
                "rappConfig",
                "wienerConfig",
                "gmpConfig",
                "piecewiseGmpConfig",
                "dohertyConfig",
                "thermalConfig",
                "parameters",
                "width",
                "outputFullScaleAmplitude",
                "parameterOverrides",
            ),
            (
                "PaModel(modelName=None, rappConfig=None, "
                "wienerConfig=None, "
                "gmpConfig=None, piecewiseGmpConfig=None, "
                "dohertyConfig=None, "
                "thermalConfig=None, "
                "parameters=None, width=None, "
                "outputFullScaleAmplitude=None, "
                "**parameterOverrides)"
            ),
        ),
        (
            Channel,
            (
                "paModel",
                "parameters",
                "width",
                "outputFullScaleAmplitude",
                "parameterOverrides",
            ),
            (
                "Channel(paModel=None, parameters=None, width=None, "
                "outputFullScaleAmplitude=None, "
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

    assert RappConfig().saturationAmplitude == 1.44
    assert WienerConfig().saturationAmplitude == 1.55
    defaultGmpConfig = GMPConfig()
    assert defaultGmpConfig.longEnvelopeMemoryDepth == 12
    assert defaultGmpConfig.longEnvelopeMemoryDecay == 0.82
    assert defaultGmpConfig.longEnvelopeMemoryCoefficient == (
        0.0215 - 0.1282j
    )

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


def CheckWifiFixedPointHeadroom() -> None:
    """Verify fixed Wi-Fi encoding preserves the floating OFDM envelope.

    Processing details:
        Algorithm: Generate identical floating and 16-bit MIMO packets, derive
        the expected common component-headroom scale, prove raw rounding never
        needs saturation, and compare decoded PAPR, normalized correlation,
        constellation references, and scale metadata with the floating source.

    Returns:
        result: None. Assertions expose fixed-boundary clipping regressions.
    """

    generationParameters = {
        "frameFormat": "EHT",
        "bandwidthMhz": 20,
        "mcs": 7,
        "numDataSymbols": 10,
        "oversampling": 4,
        "seed": 101,
        "numTransmitAntennas": 2,
        "numSpatialStreams": 2,
        "spatialMapping": "dft",
    }
    floatingWaveform = WaveGenWifi(
        **generationParameters,
        width=0,
    ).Generate()
    fixedWaveform = WaveGenWifi(
        **generationParameters,
        width=16,
    ).Generate()
    fixedFormat = FixedPoint(16)
    formatInfo = fixedFormat.GetFormatInfo()
    decodedFixedSamples = fixedFormat.DecodeComplex(
        fixedWaveform.samples
    )

    floatingPacketRms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.abs(floatingWaveform.samples) ** 2,
                    axis=1,
                )
            )
        )
    )
    assert np.isclose(
        floatingPacketRms,
        1.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    positiveComponentPeak = max(
        float(np.max(floatingWaveform.samples.real)),
        float(np.max(floatingWaveform.samples.imag)),
    )
    negativeComponentPeak = max(
        float(-np.min(floatingWaveform.samples.real)),
        float(-np.min(floatingWaveform.samples.imag)),
    )
    exactHeadroomScale = min(
        1.0,
        float(formatInfo["physicalMaximumValue"])
        / positiveComponentPeak,
        abs(float(formatInfo["physicalMinimumValue"]))
        / negativeComponentPeak,
    )
    expectedHeadroomScale = float(
        np.nextafter(exactHeadroomScale, 0.0)
    )
    metadataHeadroomScale = (
        fixedWaveform.normalizationScale
        / floatingWaveform.normalizationScale
    )
    assert exactHeadroomScale < 1.0
    assert metadataHeadroomScale == expectedHeadroomScale

    # Equality with unconstrained rounding proves EncodeComplex did not rely
    # on its saturation clip for any I/Q component. Reaching an endpoint code
    # at the unique true packet peak remains a valid, unclipped representation.
    integerScale = float(2 ** (fixedFormat.width - 1))
    scaledFloatingSamples = (
        metadataHeadroomScale * floatingWaveform.samples
    )
    unconstrainedCodes = (
        np.rint(scaledFloatingSamples.real * integerScale)
        + 1j
        * np.rint(scaledFloatingSamples.imag * integerScale)
    )
    assert np.min(unconstrainedCodes.real) >= float(
        formatInfo["minimumCode"]
    )
    assert np.max(unconstrainedCodes.real) <= float(
        formatInfo["maximumCode"]
    )
    assert np.min(unconstrainedCodes.imag) >= float(
        formatInfo["minimumCode"]
    )
    assert np.max(unconstrainedCodes.imag) <= float(
        formatInfo["maximumCode"]
    )
    assert np.array_equal(fixedWaveform.samples, unconstrainedCodes)
    assert np.allclose(
        fixedWaveform.referenceDataSymbols,
        metadataHeadroomScale
        * floatingWaveform.referenceDataSymbols,
        rtol=0.0,
        atol=2.0e-16,
    )

    decodedPacketRms = float(
        np.sqrt(
            np.mean(
                np.sum(np.abs(decodedFixedSamples) ** 2, axis=1)
            )
        )
    )
    assert np.isclose(
        decodedPacketRms,
        metadataHeadroomScale,
        rtol=0.0,
        atol=5.0e-6,
    )
    for transmitIndex in range(
        floatingWaveform.numTransmitAntennas
    ):
        floatingChain = floatingWaveform.samples[:, transmitIndex]
        fixedChain = decodedFixedSamples[:, transmitIndex]
        normalizedCorrelation = abs(
            np.vdot(floatingChain, fixedChain)
        ) / np.sqrt(
            float(np.vdot(floatingChain, floatingChain).real)
            * float(np.vdot(fixedChain, fixedChain).real)
        )
        floatingPaprDb = 10.0 * np.log10(
            np.max(np.abs(floatingChain) ** 2)
            / np.mean(np.abs(floatingChain) ** 2)
        )
        fixedPaprDb = 10.0 * np.log10(
            np.max(np.abs(fixedChain) ** 2)
            / np.mean(np.abs(fixedChain) ** 2)
        )
        assert normalizedCorrelation > 0.99999999
        assert abs(fixedPaprDb - floatingPaprDb) < 0.002

    try:
        WaveGenWifi(width=1)
    except ValueError as error:
        assert "at least two bits" in str(error)
    else:
        raise AssertionError(
            "one-bit Wi-Fi I/Q cannot represent a bipolar OFDM waveform"
        )


def CheckWifiSpectralMaskAnalysis() -> None:
    """Verify Wi-Fi mask templates, parsing modes, and spectral decisions.

    Processing details:
        Algorithm: Check every supported VHT/HE/EHT mask breakpoint, exercise
        explicit-reference, blind, metadata-assisted, and raw-assisted format
        resolution, inject deterministic upper/lower out-of-band violations,
        verify per-chain MIMO decisions, reject insufficient sample rates,
        and preserve the fixed-point public-code boundary.

    Returns:
        result: None. Assertions expose template, parser, PSD, margin, and
            result-schema regressions before a mask implementation is used.
    """

    templateBandwidths = {
        "VHT": (20, 40, 80, 160),
        "HE": (20, 40, 80, 160),
        "EHT": (20, 40, 80, 160, 320),
    }
    aliasByFormat = {
        "VHT": "11ac",
        "HE": "11ax",
        "EHT": "11be",
    }
    expectedOffsetsMhzByFormat = {
        "VHT": {
            20: (9.0, 11.0, 20.0, 30.0),
            40: (19.0, 21.0, 40.0, 60.0),
            80: (39.0, 41.0, 80.0, 120.0),
            160: (79.0, 81.0, 160.0, 240.0),
        },
        "HE": {
            20: (9.75, 10.25, 20.0, 30.0),
            40: (19.5, 20.5, 40.0, 60.0),
            80: (39.5, 40.5, 80.0, 120.0),
            160: (79.5, 80.5, 160.0, 240.0),
        },
        "EHT": {
            20: (9.75, 10.5, 20.0, 30.0),
            40: (19.5, 20.5, 40.0, 60.0),
            80: (39.5, 40.5, 80.0, 120.0),
            160: (79.5, 80.5, 160.0, 240.0),
            320: (159.5, 160.5, 320.0, 480.0),
        },
    }
    for frameFormat, bandwidthValuesMhz in templateBandwidths.items():
        for bandwidthMhz in bandwidthValuesMhz:
            template = Analysis.ResolveWifiSpectralMaskTemplate(
                frameFormat,
                bandwidthMhz,
            )
            assert isinstance(template, dict)
            assert set(template) == {
                "frameFormat",
                "bandwidthMhz",
                "templateName",
                "frequencyOffsetsHz",
                "limitsDb",
                "resolutionBandwidthHz",
                "videoBandwidthHz",
                "minimumSampleRateHz",
            }
            assert template["frameFormat"] == frameFormat
            assert template["bandwidthMhz"] == bandwidthMhz
            assert frameFormat in template["templateName"]
            assert str(bandwidthMhz) in template["templateName"]
            bandwidthHz = float(bandwidthMhz) * 1.0e6
            assert np.array_equal(
                np.asarray(template["frequencyOffsetsHz"], dtype=float),
                np.asarray(
                    expectedOffsetsMhzByFormat[frameFormat][bandwidthMhz],
                    dtype=float,
                )
                * 1.0e6,
            )
            assert tuple(template["limitsDb"]) == (
                0.0,
                -20.0,
                -28.0,
                -40.0,
            )
            assert template["resolutionBandwidthHz"] == 100.0e3
            assert template["videoBandwidthHz"] == (
                30.0e3 if frameFormat == "VHT" else 7.5e3
            )
            assert template["minimumSampleRateHz"] == (
                3.0 * bandwidthHz
                + template["resolutionBandwidthHz"]
            )

            aliasTemplate = Analysis.ResolveWifiSpectralMaskTemplate(
                aliasByFormat[frameFormat],
                bandwidthMhz,
            )
            assert aliasTemplate == template

    for invalidFormat, invalidBandwidthMhz in (
        ("VHT", 320),
        ("HE", 320),
        ("EHT", 10),
        ("unknown", 20),
    ):
        try:
            Analysis.ResolveWifiSpectralMaskTemplate(
                invalidFormat,
                invalidBandwidthMhz,
            )
        except (TypeError, ValueError) as error:
            assert "frameFormat" in str(error) or "bandwidthMhz" in str(
                error
            )
        else:
            raise AssertionError(
                "an unsupported Wi-Fi spectral-mask template was accepted"
            )

    expectedResultNames = {
        "assessmentType",
        "certificationResult",
        "frameFormat",
        "bandwidthMhz",
        "templateName",
        "passed",
        "minimumMarginDb",
        "maximumViolationDb",
        "worstFrequencyHz",
        "frequencyBinsHz",
        "maskLimitDb",
        "evaluationMask",
        "perChain",
    }
    expectedChainNames = {
        "passed",
        "minimumMarginDb",
        "maximumViolationDb",
        "worstFrequencyHz",
        "measuredPsdDb",
        "marginDb",
    }
    generatedWaveforms = {}
    for formatIndex, frameFormat in enumerate(("VHT", "HE", "EHT")):
        generatedWaveform = WaveGenWifi(
            frameFormat=frameFormat,
            bandwidthMhz=20,
            mcs=0,
            numDataSymbols=16,
            sampleRateHz=80.0e6,
            seed=710 + formatIndex,
            width=0,
        ).Generate()
        generatedWaveforms[frameFormat] = generatedWaveform
        explicitAnalysis = Analysis(
            generatedWaveform.samples,
            generatedWaveform,
            parameters={"maxSegmentLength": 4096, "width": 0},
        )
        with patch.object(
            explicitAnalysis,
            "PrepareMeasuredSignal",
            side_effect=AssertionError(
                "raw spectral-mask measurement must not use EVM resampling"
            ),
        ):
            maskResult = explicitAnalysis.MeasureWifiSpectralMask(
                generatedWaveform.samples
            )
        assert expectedResultNames.issubset(maskResult)
        assert maskResult["frameFormat"] == frameFormat
        assert maskResult["bandwidthMhz"] == 20
        assert maskResult["assessmentType"] == "relativeDbrPrecheck"
        assert maskResult["certificationResult"] is None
        assert isinstance(maskResult["passed"], bool)
        assert np.isfinite(maskResult["minimumMarginDb"])
        assert np.isfinite(maskResult["maximumViolationDb"])
        assert len(maskResult["perChain"]) == 1
        assert set(maskResult["perChain"][0]) == expectedChainNames
        assert (
            maskResult["perChain"][0]["passed"]
            is maskResult["passed"]
        )

        frequencyBinsHz = np.asarray(
            maskResult["frequencyBinsHz"], dtype=float
        )
        maskLimitDb = np.asarray(maskResult["maskLimitDb"], dtype=float)
        evaluationMask = np.asarray(
            maskResult["evaluationMask"], dtype=bool
        )
        measuredPsdDb = np.asarray(
            maskResult["perChain"][0]["measuredPsdDb"], dtype=float
        )
        marginDb = np.asarray(
            maskResult["perChain"][0]["marginDb"], dtype=float
        )
        assert frequencyBinsHz.ndim == 1
        assert frequencyBinsHz.size >= 16
        assert np.all(np.diff(frequencyBinsHz) > 0.0)
        assert maskLimitDb.shape == frequencyBinsHz.shape
        assert evaluationMask.shape == frequencyBinsHz.shape
        assert measuredPsdDb.shape == frequencyBinsHz.shape
        assert marginDb.shape == frequencyBinsHz.shape
        assert np.count_nonzero(evaluationMask) > 0
        assert np.all(np.isfinite(maskLimitDb[evaluationMask]))
        assert np.all(np.isfinite(measuredPsdDb[evaluationMask]))
        assert np.all(np.isfinite(marginDb[evaluationMask]))
        assert np.allclose(
            marginDb[evaluationMask],
            maskLimitDb[evaluationMask] - measuredPsdDb[evaluationMask],
            rtol=0.0,
            atol=1.0e-12,
        )
        assert np.isclose(
            maskResult["equivalentResolutionBandwidthHz"],
            maskResult["resolutionBandwidthHz"],
            rtol=0.0,
            atol=1.0e-8,
        )
        assert np.isclose(
            maskResult["minimumMarginDb"],
            np.min(marginDb[evaluationMask]),
            atol=1.0e-12,
        )
        assert np.isclose(
            maskResult["maximumViolationDb"],
            max(0.0, -maskResult["minimumMarginDb"]),
            atol=1.0e-12,
        )

    passingHeMask = Analysis(
        generatedWaveforms["HE"].samples,
        generatedWaveforms["HE"],
        parameters={"maxSegmentLength": 4096, "width": 0},
    ).MeasureWifiSpectralMask(generatedWaveforms["HE"].samples)
    assert passingHeMask["passed"] is True
    assert passingHeMask["minimumMarginDb"] > 0.0
    assert passingHeMask["maximumViolationDb"] == 0.0

    preparedHeAnalysis = Analysis(
        generatedWaveforms["HE"].samples,
        generatedWaveforms["HE"],
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    preparedHeSignal = preparedHeAnalysis.PrepareMeasuredSignal(
        generatedWaveforms["HE"].samples
    )
    preparedHeMask = (
        preparedHeAnalysis.CalculatePreparedWifiSpectralMask(
            preparedHeSignal
        )
    )
    assert preparedHeMask["frameFormat"] == "HE"
    assert preparedHeMask["passed"] is True
    assert np.isclose(
        preparedHeMask["minimumMarginDb"],
        passingHeMask["minimumMarginDb"],
        rtol=0.0,
        atol=0.0,
    )
    unchangedAnalysisMetrics = preparedHeAnalysis.Analyze(
        generatedWaveforms["HE"].samples
    )
    assert set(unchangedAnalysisMetrics) == {
        "snrDb",
        "evmDb",
        "evmPercent",
        "irrDb",
        "aclrLowerDb",
        "aclrUpperDb",
        "aclrWorstDb",
        "outputPowerDbm",
    }

    modeWaveform = generatedWaveforms["EHT"]
    modeMeasured = 0.83 * np.exp(1j * 0.27) * modeWaveform.samples
    blindAnalysis = Analysis(
        modeMeasured,
        parseParameters={"sampleRateHz": modeWaveform.sampleRateHz},
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    blindMask = blindAnalysis.MeasureWifiSpectralMask()
    assert blindAnalysis.GetAnalysisMode() == "blind"
    assert blindMask["frameFormat"] == "EHT"
    assert blindMask["bandwidthMhz"] == 20

    objectAssistedAnalysis = Analysis(
        modeMeasured,
        transmittedSignal=modeWaveform,
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    objectAssistedMask = (
        objectAssistedAnalysis.MeasureWifiSpectralMask()
    )
    assert objectAssistedAnalysis.GetAnalysisMode() == "transmitAssisted"
    assert objectAssistedAnalysis.GetParsedWifiFrame() is None
    assert objectAssistedMask["frameFormat"] == "EHT"
    assert objectAssistedMask["bandwidthMhz"] == 20

    rawAssistedAnalysis = Analysis(
        modeMeasured,
        transmittedSignal=modeWaveform.samples,
        sampleRateHz=modeWaveform.sampleRateHz,
        channelBandwidthHz=modeWaveform.bandwidthHz,
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    assert rawAssistedAnalysis.GetParsedWifiFrame() is None
    try:
        rawAssistedAnalysis.MeasureWifiSpectralMask()
    except ValueError as error:
        assert "wifiMaskFrameFormat" in str(error)
    else:
        raise AssertionError(
            "raw-assisted mask analysis must not parse or guess a format"
        )

    rawObjectFallbackAnalysis = Analysis(
        modeMeasured,
        transmittedSignal=modeWaveform.samples,
        sampleRateHz=modeWaveform.sampleRateHz,
        channelBandwidthHz=modeWaveform.bandwidthHz,
        parameters={
            "maxSegmentLength": 4096,
            "wifiMaskFrameFormat": "11be",
            "width": 0,
        },
    )
    rawObjectFallbackMask = (
        rawObjectFallbackAnalysis.MeasureWifiSpectralMask()
    )
    assert rawObjectFallbackMask["frameFormat"] == "EHT"
    assert rawObjectFallbackMask["bandwidthMhz"] == 20

    dataSlice = modeWaveform.fieldSlices[modeWaveform.dataFieldName]
    rawDataReference = modeWaveform.samples[dataSlice]
    rawDataMeasured = 0.91 * np.exp(-1j * 0.18) * rawDataReference
    rawFallbackAnalysis = Analysis(
        rawDataMeasured,
        transmittedSignal=rawDataReference,
        sampleRateHz=modeWaveform.sampleRateHz,
        channelBandwidthHz=modeWaveform.bandwidthHz,
        parameters={
            "maxSegmentLength": 4096,
            "wifiMaskFrameFormat": "11be",
            "width": 0,
        },
    )
    rawFallbackMask = rawFallbackAnalysis.MeasureWifiSpectralMask()
    assert rawFallbackMask["frameFormat"] == "EHT"
    assert rawFallbackMask["bandwidthMhz"] == 20

    unresolvedRawAnalysis = Analysis(
        rawDataMeasured,
        transmittedSignal=rawDataReference,
        sampleRateHz=modeWaveform.sampleRateHz,
        channelBandwidthHz=modeWaveform.bandwidthHz,
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    try:
        unresolvedRawAnalysis.MeasureWifiSpectralMask()
    except ValueError as error:
        assert "wifiMaskFrameFormat" in str(error)
    else:
        raise AssertionError(
            "descriptor-free raw-assisted mask analysis must require a "
            "frame-format fallback"
        )

    sampleIndices = np.arange(
        modeWaveform.samples.shape[0], dtype=float
    )
    upperLeakFrequencyHz = 0.75 * modeWaveform.bandwidthHz
    upperLeak = 0.20 * np.exp(
        1j
        * 2.0
        * np.pi
        * upperLeakFrequencyHz
        * sampleIndices
        / modeWaveform.sampleRateHz
    )
    upperViolation = Analysis(
        modeWaveform.samples,
        modeWaveform,
        parameters={"maxSegmentLength": 4096, "width": 0},
    ).MeasureWifiSpectralMask(modeWaveform.samples + upperLeak)
    assert upperViolation["passed"] is False
    assert upperViolation["minimumMarginDb"] < 0.0
    assert upperViolation["maximumViolationDb"] > 0.0
    assert upperViolation["worstFrequencyHz"] > 0.0

    lowerLeak = 0.20 * np.exp(
        -1j
        * 2.0
        * np.pi
        * upperLeakFrequencyHz
        * sampleIndices
        / modeWaveform.sampleRateHz
    )
    lowerViolation = Analysis(
        modeWaveform.samples,
        modeWaveform,
        parameters={"maxSegmentLength": 4096, "width": 0},
    ).MeasureWifiSpectralMask(modeWaveform.samples + lowerLeak)
    assert lowerViolation["passed"] is False
    assert lowerViolation["minimumMarginDb"] < 0.0
    assert lowerViolation["maximumViolationDb"] > 0.0
    assert lowerViolation["worstFrequencyHz"] < 0.0

    mimoWaveform = WaveGenWifi(
        frameFormat="HE",
        bandwidthMhz=20,
        mcs=0,
        numDataSymbols=16,
        sampleRateHz=80.0e6,
        numTransmitAntennas=2,
        numSpatialStreams=2,
        spatialMapping="dft",
        seed=719,
        width=0,
    ).Generate()
    mimoAnalysis = Analysis(
        mimoWaveform.samples,
        mimoWaveform,
        parameters={"maxSegmentLength": 4096, "width": 0},
    )
    mimoSamples = mimoWaveform.samples.copy()
    mimoSampleIndices = np.arange(mimoSamples.shape[0], dtype=float)
    mimoSamples[:, 1] += 0.20 * np.exp(
        1j
        * 2.0
        * np.pi
        * 0.75
        * mimoWaveform.bandwidthHz
        * mimoSampleIndices
        / mimoWaveform.sampleRateHz
    )
    with patch(
        "inc.lib.Analysis.AveragePeriodogram",
        wraps=AveragePeriodogram,
    ) as periodogramMock:
        mimoResult = mimoAnalysis.MeasureWifiSpectralMask(mimoSamples)
        assert periodogramMock.call_count == 2
    assert len(mimoResult["perChain"]) == 2
    assert mimoResult["perChain"][0]["passed"] is True
    assert mimoResult["perChain"][1]["passed"] is False
    assert mimoResult["passed"] is False
    assert np.isclose(
        mimoResult["minimumMarginDb"],
        min(
            chainResult["minimumMarginDb"]
            for chainResult in mimoResult["perChain"]
        ),
        atol=1.0e-12,
    )

    lowRateWaveform = WaveGenWifi(
        frameFormat="VHT",
        bandwidthMhz=20,
        mcs=0,
        numDataSymbols=4,
        sampleRateHz=40.0e6,
        seed=720,
        width=0,
    ).Generate()
    lowRateAnalysis = Analysis(
        lowRateWaveform.samples,
        lowRateWaveform,
        width=0,
    )
    try:
        lowRateAnalysis.MeasureWifiSpectralMask(lowRateWaveform.samples)
    except ValueError as error:
        assert "sampleRateHz" in str(error)
        assert "spectral mask" in str(error).lower()
    else:
        raise AssertionError(
            "mask analysis accepted a sample rate below its outer offset"
        )

    minimumMaskSampleRateHz = 60.1e6
    exactRateRandom = np.random.default_rng(722)
    exactRateReference = (
        exactRateRandom.normal(size=8192)
        + 1j * exactRateRandom.normal(size=8192)
    )
    exactRateAnalysis = Analysis(
        exactRateReference,
        transmittedSignal=exactRateReference,
        sampleRateHz=minimumMaskSampleRateHz,
        channelBandwidthHz=20.0e6,
        parameters={
            "maxSegmentLength": 4096,
            "wifiMaskFrameFormat": "EHT",
            "width": 0,
        },
    )
    exactRateResult = exactRateAnalysis.MeasureWifiSpectralMask()
    assert exactRateResult["sampleRateHz"] == minimumMaskSampleRateHz
    assert (
        np.isclose(
            exactRateResult["equivalentResolutionBandwidthHz"],
            exactRateResult["resolutionBandwidthHz"],
            rtol=0.0,
            atol=1.0e-8,
        )
    )
    flatFrequencyBinsHz = np.fft.fftshift(
        np.fft.fftfreq(1024, d=1.0 / minimumMaskSampleRateHz)
    )
    with patch(
        "inc.lib.Analysis.AveragePeriodogram",
        return_value=(
            flatFrequencyBinsHz,
            np.ones(flatFrequencyBinsHz.size, dtype=float),
        ),
    ):
        flatSpectrumResult = exactRateAnalysis.MeasureWifiSpectralMask()
    assert np.allclose(
        flatSpectrumResult["perChain"][0]["measuredPsdDb"],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    fixedWaveform = WaveGenWifi(
        frameFormat="VHT",
        bandwidthMhz=20,
        mcs=0,
        numDataSymbols=16,
        sampleRateHz=80.0e6,
        seed=721,
        width=16,
    ).Generate()
    fixedAnalysis = Analysis(
        fixedWaveform.samples,
        fixedWaveform,
        parameters={"maxSegmentLength": 4096, "width": 16},
    )
    fixedResult = fixedAnalysis.MeasureWifiSpectralMask(
        fixedWaveform.samples
    )
    assert fixedResult["frameFormat"] == "VHT"
    assert fixedResult["bandwidthMhz"] == 20
    assert isinstance(fixedResult["passed"], bool)
    assert np.isfinite(fixedResult["minimumMarginDb"])
    assert np.isfinite(fixedResult["maximumViolationDb"])
    assert np.array_equal(
        fixedWaveform.samples.real,
        np.rint(fixedWaveform.samples.real),
    )
    assert np.array_equal(
        fixedWaveform.samples.imag,
        np.rint(fixedWaveform.samples.imag),
    )


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
        assert len(mimoMetrics["irrDbPerChain"]) == transmitCount
        assert max(mimoMetrics["irrDbPerChain"]) < -200.0
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
            "irrDb",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
            "outputPowerDbm",
        }
        assert not hasattr(metrics, "ToDict")
        assert metrics["snrDb"] > 250.0
        assert metrics["evmDb"] < -250.0
        assert metrics["evmPercent"] < 1e-10
        assert metrics["irrDb"] < -200.0
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

    iqReference = (
        np.random.default_rng(404).standard_normal(4096)
        + 1j * np.random.default_rng(405).standard_normal(4096)
    )
    iqReference /= np.sqrt(np.mean(np.abs(iqReference) ** 2))
    imageCoefficient = 0.05 * np.exp(1j * 0.3)
    iqMeasured = iqReference + imageCoefficient * np.conj(iqReference)
    iqAnalysis = Analysis(
        iqMeasured,
        parameters={
            "width": 0,
            "sampleRateHz": 80.0e6,
        },
        transmittedSignal=iqReference,
    )
    iqMetrics = iqAnalysis.Analyze()
    irrMeasurement = iqAnalysis.MeasureIrr()
    expectedIrrDb = 20.0 * np.log10(np.abs(imageCoefficient))
    assert abs(iqMetrics["irrDb"] - expectedIrrDb) < 0.1
    assert abs(irrMeasurement["irrDb"] - expectedIrrDb) < 0.1
    assert abs(iqAnalysis.CalculateIrr() - expectedIrrDb) < 0.1
    assert np.isclose(
        irrMeasurement["imageAmplitudeRatio"],
        np.abs(imageCoefficient),
        atol=1.0e-5,
    )
    assert irrMeasurement["irrDbPerChain"] == (
        irrMeasurement["irrDb"],
    )
    assert (
        irrMeasurement["regressionConditionNumberPerChain"][0]
        < 1.1
    )
    assert irrMeasurement["residualPowerRatio"] < 1.0e-5
    assert set(irrMeasurement) == {
        "irrDb",
        "irrDbPerChain",
        "desiredCoefficientPower",
        "imageCoefficientPower",
        "imageAmplitudeRatio",
        "directCoefficientRealPerChain",
        "directCoefficientImagPerChain",
        "imageCoefficientRealPerChain",
        "imageCoefficientImagPerChain",
        "residualPowerRatio",
        "regressionConditionNumberPerChain",
    }


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
    decodedFixedBurst = FixedPoint(
        16, fixedPaModel.outputFullScaleAmplitude
    ).DecodeComplex(
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
            "outputFullScaleAmplitude": (
                fixedPaModel.outputFullScaleAmplitude
            ),
        },
    ).Analyze()
    assert abs(fixedBurstMetrics["outputPowerDbm"] - 22.0) < 0.01
    fixedReplayedBurst = fixedPaModel.Process(fixedCalibratedInput)
    assert np.array_equal(fixedReplayedBurst, fixedCalibratedBurst)

    fixedMimoPaModel = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {"modelName": "wiener"},
                {"modelName": "gmp"},
            ),
            "width": 16,
        }
    )
    fixedMimoCalibration = PowerCalibration(
        paModel=fixedMimoPaModel,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbmPerChain": (18.0, 20.0),
            "calibrationToleranceDb": 0.25,
            "maximumCalibrationIterations": 60,
            "width": 16,
        },
    )
    fixedMimoInput = np.column_stack(
        (fixedInputBurst, fixedInputBurst)
    )
    fixedMimoCalibratedInput = fixedMimoCalibration.Calibrate(
        fixedMimoInput
    )
    fixedMimoAcceptedOutput = fixedMimoCalibration.GetLastPaOutput()
    fixedMimoReplayedOutput = fixedMimoPaModel.Process(
        fixedMimoCalibratedInput
    )
    assert np.array_equal(
        fixedMimoReplayedOutput, fixedMimoAcceptedOutput
    )
    fixedMimoMetrics = fixedMimoCalibration.GetLastCalibrationMetrics()
    assert np.all(
        np.abs(
            np.asarray(
                fixedMimoMetrics["measuredOutputPowerDbmPerChain"]
            )
            - np.asarray((18.0, 20.0))
        )
        <= 0.25
    )

    # A fixed-only third-party plant without a post-decode drive interface can
    # still have a genuinely unreachable target. The failure must preserve a
    # useful best measurement instead of merely reporting an iteration limit.
    class UnreachableFixedPa:
        """Return a deterministic 10 dBm active waveform for every trial."""

        def __init__(self) -> None:
            """Initialize the synthetic PA at the shared 16-bit boundary.

            Processing details:
                Algorithm: Store the external word width used by
                ``PowerCalibration.SetPaModel`` compatibility validation.

            Returns:
                result: None. The constant-output plant is ready.
            """

            self.width = 16

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Return a phase-preserving active burst at exactly 10 dBm.

            Processing details:
                Algorithm: Decode the public trial, retain its active support
                and sample phase, replace every active magnitude by the
                normalized RMS for 10 dBm under a 25 dBm full-scale mapping,
                and encode the result once.

            Args:
                inputSignal: Public 16-bit trial waveform.

            Returns:
                result: Public fixed-point waveform whose active RMS is 10 dBm.
            """

            fixedFormat = FixedPoint(self.width)
            floatingInput = fixedFormat.DecodeComplex(inputSignal)
            activeSamples = np.abs(floatingInput) > np.finfo(float).tiny
            floatingOutput = np.zeros_like(
                floatingInput, dtype=np.complex128
            )
            outputRms = np.power(10.0, (10.0 - 25.0) / 20.0)
            floatingOutput[activeSamples] = outputRms * np.exp(
                1j * np.angle(floatingInput[activeSamples])
            )
            return fixedFormat.EncodeComplex(floatingOutput)

    unreachableCalibration = PowerCalibration(
        paModel=UnreachableFixedPa(),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 20.0,
            "calibrationToleranceDb": 0.1,
            "maximumCalibrationIterations": 6,
            "width": 16,
        },
    )
    try:
        unreachableCalibration.Calibrate(fixedInputBurst)
    except RuntimeError as error:
        failureMessage = str(error)
        assert "did not converge" in failureMessage
        assert "target" in failureMessage
        assert "best measured" in failureMessage
    else:
        raise AssertionError("an unreachable fixed-point target converged")
    failureMetrics = unreachableCalibration.GetLastCalibrationMetrics()
    assert failureMetrics["converged"] is False
    assert failureMetrics["targetOutputPowerDbmPerChain"] == (20.0,)
    assert np.isclose(
        failureMetrics["measuredOutputPowerDbmPerChain"][0],
        10.0,
        atol=0.15,
    )
    assert np.isclose(
        failureMetrics["errorDbPerChain"][0], 10.0, atol=0.15
    )
    assert 1 <= failureMetrics["iterationCount"] <= 6
    assert "analogDriveDbPerChain" not in failureMetrics

    # Repeated legacy codes can reflect an ordinary quantization step rather
    # than digital full scale. A deliberately tiny learning rate must exhaust
    # its configured trial budget without reporting a false clipping plateau.
    class LegacyFixedLinearPa:
        """Apply a fixed linear gain without a post-decode drive protocol."""

        def __init__(self, gain: float = 0.8) -> None:
            """Initialize the legacy adapter at an 8-bit public boundary.

            Processing details:
                Algorithm: Store the public width used by compatibility
                validation and a mutable scalar plant gain while intentionally
                omitting the paired calibration drive methods.

            Args:
                gain: Finite linear complex-envelope amplitude gain.

            Returns:
                result: None. The linear legacy plant is ready.
            """

            self.width = 8
            self.gain = float(gain)

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Apply a finite linear gain across the fixed-point boundary.

            Processing details:
                Algorithm: Decode signed 8-bit I/Q codes, multiply every
                complex sample by 0.8, and encode the output once.

            Args:
                inputSignal: Public 8-bit complex I/Q codes.

            Returns:
                result: Public fixed-point output from the reachable plant.
            """

            fixedFormat = FixedPoint(self.width)
            floatingInput = fixedFormat.DecodeComplex(inputSignal)
            return fixedFormat.EncodeComplex(self.gain * floatingInput)

    legacyCalibration = PowerCalibration(
        paModel=LegacyFixedLinearPa(),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 20.0,
            "calibrationToleranceDb": 0.01,
            "maximumCalibrationIterations": 6,
            "calibrationLearningRate": 1.0e-4,
            "width": 8,
        },
    )
    legacyInputBurst = FixedPoint(8).EncodeComplex(paddedBurst / 3.0)
    try:
        legacyCalibration.Calibrate(legacyInputBurst)
    except RuntimeError:
        legacyMetrics = legacyCalibration.GetLastCalibrationMetrics()
        assert legacyMetrics["iterationCount"] == 6
        assert "every nonzero" not in legacyMetrics["failureReason"]
        assert "analogDriveDbPerChain" not in legacyMetrics
    else:
        raise AssertionError("the tiny-step legacy test unexpectedly converged")

    asymmetricLegacyCalibration = PowerCalibration(
        paModel=LegacyFixedLinearPa(),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.01,
            "maximumCalibrationIterations": 6,
            "calibrationLearningRate": 1.0e-4,
            "width": 8,
        },
    )
    asymmetricLegacyInput = FixedPoint(8).EncodeComplex(
        np.full(128, 0.5 + 0.05j, dtype=np.complex128)
    )
    try:
        asymmetricLegacyCalibration.Calibrate(asymmetricLegacyInput)
    except RuntimeError:
        asymmetricLegacyMetrics = (
            asymmetricLegacyCalibration.GetLastCalibrationMetrics()
        )
        assert asymmetricLegacyMetrics["iterationCount"] == 6
        assert "every nonzero" not in (
            asymmetricLegacyMetrics["failureReason"]
        )
    else:
        raise AssertionError(
            "the asymmetric tiny-step legacy test unexpectedly converged"
        )

    # A terminal codeword is only a blocker when output power is too low.
    # If a changed plant makes the same codeword too powerful, calibration must
    # reduce drive, leave the rail, and find the lower reachable operating point.
    mutableLegacyPa = LegacyFixedLinearPa(gain=0.7)
    railEscapeCalibration = PowerCalibration(
        paModel=mutableLegacyPa,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "outputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.25,
            "maximumCalibrationIterations": 60,
            "calibrationLearningRate": 0.8,
            "width": 8,
        },
    )
    terminalInput = FixedPoint(8).EncodeComplex(
        np.full(128, 0.5 + 0.5j, dtype=np.complex128)
    )
    railEscapeCalibration.Calibrate(terminalInput)
    mutableLegacyPa.gain = 0.9
    railEscapeCalibration.UpdateParameters(
        calibrationToleranceDb=0.05,
        maximumCalibrationIterations=500,
        calibrationLearningRate=0.01,
    )
    escapedRailInput = railEscapeCalibration.Calibrate(terminalInput)
    escapedRailMetrics = railEscapeCalibration.GetLastCalibrationMetrics()
    assert escapedRailMetrics["converged"] is True
    assert abs(
        escapedRailMetrics["measuredOutputPowerDbmPerChain"][0] - 25.0
    ) <= 0.05
    assert np.max(np.abs(escapedRailInput.real)) < 127.0
    assert np.max(np.abs(escapedRailInput.imag)) < 127.0

    for invalidPowerParameters in (
        {"width": 1},
        {"width": 2, "calibrationDigitalHeadroomDb": 60.0},
        {"width": 8, "calibrationDigitalHeadroomDb": 49.0},
    ):
        try:
            PowerCalibration(parameters=invalidPowerParameters)
        except ValueError as error:
            assert "Allowed" in str(error) or "at least 2" in str(error)
        else:
            raise AssertionError(
                "an unusable fixed-point calibration format was accepted"
            )

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


def CheckIlcPowerOperatingPoints() -> None:
    """Verify that fixed-point ILC preserves calibrated power semantics.

    Processing details:
        Algorithm: Calibrate one common legal 16-bit Wi-Fi waveform to four
        conducted powers through hidden post-DAC drive, run a short clean ILC
        at every operating point, require every stored chOut to remain at its
        own power, and select the strict chOut-EVM best round. Repeat the
        three-power sweep without quantization to check intrinsic EVM order,
        then check two independently calibrated MIMO chains, an IQ wrapper,
        and steady-state thermal Channel recalibration and period accounting.

    Returns:
        result: None. Assertions enforce power, reference-plane, and EVM
        ordering contracts across calibrated ILC operating points.
    """

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        seed=91,
        width=16,
    ).Generate()
    targetPowerDbmValues = (1.0, 10.0, 16.0, 20.0)
    publicReferences = []
    analogDriveDbValues = []
    fixedBaselineEvmDbValues = []
    rawFloatingOutputs = []

    for targetPowerDbm in targetPowerDbmValues:
        paModel = PaModel(modelName="gmp", width=16)
        powerCalibration = PowerCalibration(
            paModel=paModel,
            parameters={
                "outputPowerDbm": targetPowerDbm,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.05,
                "maximumCalibrationIterations": 60,
                "width": 16,
            },
        )
        referenceSignal = powerCalibration.Calibrate(waveform.samples)
        publicReferences.append(referenceSignal.copy())
        calibrationMetrics = powerCalibration.GetLastCalibrationMetrics()
        analogDriveDbValues.append(
            float(calibrationMetrics["analogDriveDbPerChain"][0])
        )
        assert abs(
            calibrationMetrics["measuredOutputPowerDbmPerChain"][0]
            - targetPowerDbm
        ) <= 0.05

        resultAnalysis = Analysis(
            referenceSignal,
            waveform,
            parameters={
                "maximumOutputPowerDbm": 25.0,
                "width": 16,
                "outputFullScaleAmplitude": (
                    paModel.outputFullScaleAmplitude
                ),
            },
        )
        baselinePaOutput = powerCalibration.GetLastPaOutput()
        baselineMetrics = resultAnalysis.Analyze(baselinePaOutput)
        fixedBaselineEvmDbValues.append(float(baselineMetrics["evmDb"]))
        assert abs(
            baselineMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        automaticScaleAnalysis = Analysis(
            referenceSignal,
            waveform,
            parameters={
                "maximumOutputPowerDbm": 25.0,
                "width": 16,
            },
        )
        automaticScaleMetrics = automaticScaleAnalysis.Analyze(
            baselinePaOutput
        )
        assert abs(
            automaticScaleMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        if targetPowerDbm == 20.0:
            blindAutomaticScaleMetrics = Analysis(
                baselinePaOutput
            ).Analyze()
            assert abs(
                blindAutomaticScaleMetrics["outputPowerDbm"]
                - targetPowerDbm
            ) <= 0.10
            explicitLegacyScaleMetrics = Analysis(
                referenceSignal,
                waveform,
                parameters={
                    "maximumOutputPowerDbm": 25.0,
                    "width": 16,
                    "outputFullScaleAmplitude": 1.0,
                },
            ).Analyze(baselinePaOutput)
            assert np.isclose(
                explicitLegacyScaleMetrics["outputPowerDbm"],
                baselineMetrics["outputPowerDbm"]
                - 20.0 * np.log10(2.0),
                atol=0.01,
            )

        # Public floating processing must retain the hidden post-DAC drive just
        # like public fixed processing. The explicit raw kernel is reserved for
        # calibration/Channel code that has already applied that drive.
        inputFormat = FixedPoint(16)
        outputFormat = FixedPoint(
            16, paModel.outputFullScaleAmplitude
        )
        normalizedReference = inputFormat.DecodeComplex(referenceSignal)
        floatingPaOutput = paModel.ProcessFloating(normalizedReference)
        rawFloatingOutputs.append(
            paModel.ProcessRawFloating(normalizedReference)
        )
        encodedFloatingPaOutput = outputFormat.EncodeComplex(
            floatingPaOutput
        )
        assert np.array_equal(encodedFloatingPaOutput, baselinePaOutput)
        assert np.array_equal(
            encodedFloatingPaOutput,
            paModel.Process(referenceSignal),
        )
        floatingMetrics = resultAnalysis.Analyze(encodedFloatingPaOutput)
        assert abs(
            floatingMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        assert np.isclose(
            floatingMetrics["evmDb"],
            baselineMetrics["evmDb"],
            rtol=0.0,
            atol=1.0e-12,
        )
        if targetPowerDbm == 20.0:
            baselineFormatInfo = FixedPoint(
                16, paModel.outputFullScaleAmplitude
            ).GetFormatInfo()
            assert not np.any(
                np.abs(baselinePaOutput.real)
                >= float(baselineFormatInfo["maximumCode"])
            )
            assert not np.any(
                np.abs(baselinePaOutput.imag)
                >= float(baselineFormatInfo["maximumCode"])
            )

        ilcResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            ILCConfig(
                numIterations=3,
                learningRate=0.15,
                regularization=1.0e-3,
                maxAmplitude=2.0,
                randomSeed=1091,
            ),
        )
        analyzedHistory = resultAnalysis.AnalyzeIlcHistory(
            ilcResult.history
        )
        assert len(analyzedHistory.history) == 3
        iterationPowerDbm = np.asarray(
            [
                iterationRecord.outputPowerDbm
                for iterationRecord in analyzedHistory.history
            ],
            dtype=float,
        )
        assert np.all(
            np.abs(iterationPowerDbm - targetPowerDbm) <= 0.35
        )
        iterationEvmDb = np.asarray(
            [
                iterationRecord.evmDb
                for iterationRecord in analyzedHistory.history
            ],
            dtype=float,
        )
        expectedBestIndex = int(np.argmin(iterationEvmDb))
        assert analyzedHistory.bestIteration == expectedBestIndex + 1
        assert np.array_equal(
            analyzedHistory.bestInputSignal,
            ilcResult.history[expectedBestIndex].inputSignal,
        )
        assert np.array_equal(
            analyzedHistory.bestOutputSignal,
            ilcResult.history[expectedBestIndex].outputSignal,
        )

        powerCalibration.Calibrate(analyzedHistory.bestInputSignal)
        selectedOutput = powerCalibration.GetLastPaOutput()
        selectedMetrics = resultAnalysis.Analyze(selectedOutput)
        assert abs(
            selectedMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        assert selectedMetrics["evmDb"] <= baselineMetrics["evmDb"] + 0.15

        if targetPowerDbm == 20.0:
            publicPaOutput = powerCalibration.GetLastPaOutput()
            formatInfo = FixedPoint(
                16, paModel.outputFullScaleAmplitude
            ).GetFormatInfo()
            assert not np.any(
                np.abs(publicPaOutput.real)
                >= float(formatInfo["maximumCode"])
            )
            assert not np.any(
                np.abs(publicPaOutput.imag)
                >= float(formatInfo["maximumCode"])
            )

    # With fixed-point digital headroom, all four conducted powers use the
    # same legal DAC codes. Their distinct physical operating points reside in
    # the committed post-decode analog drive and must survive ILC adaptation.
    assert all(
        np.array_equal(publicReferences[0], publicReference)
        for publicReference in publicReferences[1:]
    )
    assert all(
        np.array_equal(rawFloatingOutputs[0], rawFloatingOutput)
        for rawFloatingOutput in rawFloatingOutputs[1:]
    )
    assert np.all(np.diff(np.asarray(analogDriveDbValues)) > 3.0)
    assert np.all(
        np.abs(
            np.asarray(fixedBaselineEvmDbValues, dtype=float)
            - np.asarray((-51.5, -49.0, -40.4, -32.6), dtype=float)
        )
        <= 1.5
    )

    # The 2.0 default is chosen for the requested low-/mid-/20 dBm EVM
    # resolution. Near the 25 dBm rated endpoint, a caller can explicitly
    # trade one additional bit of resolution for another 6.02 dB of peak
    # observation headroom.
    ratedHeadroomPa = PaModel(
        modelName="gmp",
        width=16,
        outputFullScaleAmplitude=4.0,
    )
    ratedHeadroomCalibration = PowerCalibration(
        paModel=ratedHeadroomPa,
        parameters={
            "outputPowerDbm": 25.0,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.15,
            "maximumCalibrationIterations": 60,
            "width": 16,
        },
    )
    ratedReference = ratedHeadroomCalibration.Calibrate(
        waveform.samples
    )
    ratedOutput = ratedHeadroomCalibration.GetLastPaOutput()
    ratedMetrics = Analysis(
        ratedReference,
        waveform,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 16,
            "outputFullScaleAmplitude": 4.0,
        },
    ).Analyze(ratedOutput)
    assert abs(ratedMetrics["outputPowerDbm"] - 25.0) <= 0.15
    automaticRatedMetrics = Analysis(
        ratedReference,
        waveform,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 16,
        },
    ).Analyze(ratedOutput)
    assert abs(
        automaticRatedMetrics["outputPowerDbm"] - 25.0
    ) <= 0.15
    ratedFormatInfo = FixedPoint(16, 4.0).GetFormatInfo()
    assert not np.any(
        np.abs(ratedOutput.real)
        >= float(ratedFormatInfo["maximumCode"])
    )
    assert not np.any(
        np.abs(ratedOutput.imag)
        >= float(ratedFormatInfo["maximumCode"])
    )

    # Power ordering is an intrinsic floating-point PA expectation. Do not
    # impose it on the fixed-point checks above: a quantization floor can
    # legitimately dominate a sufficiently backed-off waveform.
    floatingWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        seed=91,
        width=0,
    ).Generate()
    floatingBaselineEvmDbValues = []
    floatingFinalEvmDbValues = []
    floatingIterationEvmDbRows = []
    intrinsicPowerDbmValues = (1.0, 16.0, 20.0)
    for targetPowerDbm in intrinsicPowerDbmValues:
        floatingPaModel = PaModel(modelName="gmp", width=0)
        floatingPowerCalibration = PowerCalibration(
            paModel=floatingPaModel,
            parameters={
                "outputPowerDbm": targetPowerDbm,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.05,
                "width": 0,
            },
        )
        floatingReference = floatingPowerCalibration.Calibrate(
            floatingWaveform.samples
        )
        floatingAnalysis = Analysis(
            floatingReference,
            floatingWaveform,
            parameters={
                "maximumOutputPowerDbm": 25.0,
                "width": 0,
            },
        )
        floatingBaselineMetrics = floatingAnalysis.Analyze(
            floatingPowerCalibration.GetLastPaOutput()
        )
        floatingBaselineEvmDbValues.append(
            float(floatingBaselineMetrics["evmDb"])
        )
        floatingIlcResult = RunFrequencyDomainIlc(
            floatingReference,
            floatingPaModel,
            floatingWaveform.sampleRateHz,
            floatingWaveform.bandwidthHz,
            ILCConfig(
                numIterations=3,
                learningRate=0.15,
                regularization=1.0e-3,
                maxAmplitude=2.0,
                randomSeed=1391,
            ),
        )
        floatingHistory = floatingAnalysis.AnalyzeIlcHistory(
            floatingIlcResult.history
        )
        floatingIterationEvmDbRows.append(
            np.asarray(
                [
                    iterationRecord.evmDb
                    for iterationRecord in floatingHistory.history
                ],
                dtype=float,
            )
        )
        assert np.all(
            np.abs(
                np.asarray(
                    [
                        iterationRecord.outputPowerDbm
                        for iterationRecord in floatingHistory.history
                    ],
                    dtype=float,
                )
                - targetPowerDbm
            )
            <= 0.35
        )
        floatingPowerCalibration.Calibrate(
            floatingHistory.bestInputSignal
        )
        floatingFinalMetrics = floatingAnalysis.Analyze(
            floatingPowerCalibration.GetLastPaOutput()
        )
        floatingFinalEvmDbValues.append(
            float(floatingFinalMetrics["evmDb"])
        )
        assert abs(
            floatingFinalMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        assert (
            floatingFinalMetrics["evmDb"]
            <= floatingBaselineMetrics["evmDb"] + 0.15
        )

    floatingBaselineEvmDb = np.asarray(
        floatingBaselineEvmDbValues, dtype=float
    )
    floatingIterationEvmDbMatrix = np.vstack(
        floatingIterationEvmDbRows
    )
    floatingFinalEvmDb = np.asarray(
        floatingFinalEvmDbValues, dtype=float
    )
    # No receiver/feedback noise, thermal drift, or fixed-point quantization
    # is present here. At every measured round and after equal-power replay,
    # lower output power must therefore remain no worse for the default
    # monotonic GMP plant. More negative EVM dB is better.
    assert np.all(np.diff(floatingBaselineEvmDb) >= -0.25)
    assert np.all(
        np.diff(floatingIterationEvmDbMatrix, axis=0) >= -0.25
    )
    assert np.all(np.diff(floatingFinalEvmDb) >= -0.25)
    # The noiseless default plant must expose a materially different
    # nonlinear operating point at every requested power. The stronger
    # deterministic long-envelope residual intentionally targets roughly
    # -51.5/-40.4/-32.6 dB at 1/16/20 dBm for this noiseless EHT frame.
    expectedBaselineEvmDb = np.asarray(
        (-51.5, -40.4, -32.6), dtype=float
    )
    assert np.all(
        np.abs(floatingBaselineEvmDb - expectedBaselineEvmDb) <= 1.5
    )
    assert np.all(np.diff(floatingBaselineEvmDb) > 2.0)
    assert floatingBaselineEvmDb[-1] - floatingBaselineEvmDb[0] > 8.0
    assert np.all(np.diff(floatingFinalEvmDb) > 2.0)
    assert floatingFinalEvmDb[-1] - floatingFinalEvmDb[0] > 8.0

    # Doherty is intentionally branch-aware and may outperform the ordinary
    # GMP at some powers, but its default must not collapse into a much worse
    # plant. Compare equal-power, identical-frame baselines and retain wide
    # margins around the expected power trend instead of pinning exact EVMs.
    dohertyBaselineEvmDbValues = []
    for targetPowerDbm in intrinsicPowerDbmValues:
        dohertyPaModel = PaModel(modelName="doherty", width=0)
        dohertyPowerCalibration = PowerCalibration(
            paModel=dohertyPaModel,
            parameters={
                "outputPowerDbm": targetPowerDbm,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.05,
                "width": 0,
            },
        )
        dohertyReference = dohertyPowerCalibration.Calibrate(
            floatingWaveform.samples
        )
        dohertyMetrics = Analysis(
            dohertyReference,
            floatingWaveform,
            parameters={
                "maximumOutputPowerDbm": 25.0,
                "width": 0,
            },
        ).Analyze(dohertyPowerCalibration.GetLastPaOutput())
        assert abs(
            dohertyMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.10
        dohertyBaselineEvmDbValues.append(
            float(dohertyMetrics["evmDb"])
        )
    dohertyBaselineEvmDb = np.asarray(
        dohertyBaselineEvmDbValues, dtype=float
    )
    assert np.all(np.diff(dohertyBaselineEvmDb) > 2.0)
    assert dohertyBaselineEvmDb[-1] - dohertyBaselineEvmDb[0] > 12.0
    assert dohertyBaselineEvmDb[-1] < -24.0

    mimoInput = np.column_stack((waveform.samples, waveform.samples))
    mimoPaModel = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {"modelName": "gmp"},
                {"modelName": "gmp"},
            ),
            "width": 16,
        }
    )
    mimoTargetPowerDbm = (10.0, 20.0)
    mimoPowerCalibration = PowerCalibration(
        paModel=mimoPaModel,
        parameters={
            "outputPowerDbm": mimoTargetPowerDbm[0],
            "outputPowerDbmPerChain": mimoTargetPowerDbm,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "maximumCalibrationIterations": 60,
            "width": 16,
        },
    )
    mimoReference = mimoPowerCalibration.Calibrate(mimoInput)
    mimoInputFormat = FixedPoint(16)
    mimoOutputFormat = FixedPoint(
        16, mimoPaModel.outputFullScaleAmplitude
    )
    mimoFloatingOutput = mimoPaModel.ProcessFloating(
        mimoInputFormat.DecodeComplex(mimoReference)
    )
    encodedMimoFloatingOutput = mimoOutputFormat.EncodeComplex(
        mimoFloatingOutput
    )
    assert np.array_equal(
        encodedMimoFloatingOutput,
        mimoPaModel.Process(mimoReference),
    )
    assert np.array_equal(
        encodedMimoFloatingOutput,
        mimoPowerCalibration.GetLastPaOutput(),
    )
    mimoIlcResult = RunMimoFrequencyDomainIlc(
        mimoReference,
        mimoPaModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(
            numIterations=1,
            learningRate=0.15,
            regularization=1.0e-3,
            maxAmplitude=2.0,
            randomSeed=1191,
        ),
    )
    for chainIndex, targetPowerDbm in enumerate(mimoTargetPowerDbm):
        chainAnalysis = Analysis(
            mimoReference[:, chainIndex],
            waveform,
            parameters={
                "maximumOutputPowerDbm": 25.0,
                "width": 16,
                "outputFullScaleAmplitude": (
                    mimoPaModel.outputFullScaleAmplitude
                ),
            },
        )
        chainMetrics = chainAnalysis.Analyze(
            mimoIlcResult.chainResults[chainIndex]
            .history[0]
            .outputSignal
        )
        assert abs(
            chainMetrics["outputPowerDbm"] - targetPowerDbm
        ) <= 0.35


    wrappedPaModel = PaModel(modelName="gmp", width=16)
    iqWrappedPaModel = IQImbalancePA(
        wrappedPaModel,
        directCoefficient=0.99 + 0.01j,
        imageCoefficient=0.02 - 0.01j,
    )
    originalWrappedDriveCommit = wrappedPaModel.SetCalibrationDriveDb
    with patch.object(
        wrappedPaModel,
        "SetCalibrationDriveDb",
        wraps=originalWrappedDriveCommit,
    ) as wrappedDriveCommit:
        wrappedPowerCalibration = PowerCalibration(
            paModel=iqWrappedPaModel,
            parameters={
                "outputPowerDbm": 10.0,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.05,
                "width": 16,
            },
        )
        wrappedReference = wrappedPowerCalibration.Calibrate(
            waveform.samples
        )
    wrappedDriveCommit.assert_called_once()
    committedWrappedDriveDb = float(
        wrappedDriveCommit.call_args.args[0][0]
    )
    assert abs(committedWrappedDriveDb) > 1.0
    fixedFormatInfo = FixedPoint(16).GetFormatInfo()
    digitalHeadroomLimit = (
        float(fixedFormatInfo["maximumCode"])
        * 10.0 ** (-6.0 / 20.0)
        + 1.0
    )
    assert np.max(np.abs(wrappedReference.real)) <= digitalHeadroomLimit
    assert np.max(np.abs(wrappedReference.imag)) <= digitalHeadroomLimit
    wrappedInputFormat = FixedPoint(16)
    wrappedOutputFormat = FixedPoint(
        16, iqWrappedPaModel.outputFullScaleAmplitude
    )
    wrappedFloatingOutput = iqWrappedPaModel.ProcessFloating(
        wrappedInputFormat.DecodeComplex(wrappedReference)
    )
    encodedWrappedFloatingOutput = wrappedOutputFormat.EncodeComplex(
        wrappedFloatingOutput
    )
    assert np.array_equal(
        encodedWrappedFloatingOutput,
        iqWrappedPaModel.Process(wrappedReference),
    )
    assert np.array_equal(
        encodedWrappedFloatingOutput,
        wrappedPowerCalibration.GetLastPaOutput(),
    )
    assert not np.allclose(
        wrappedFloatingOutput,
        iqWrappedPaModel.ProcessRawFloating(
            wrappedInputFormat.DecodeComplex(wrappedReference)
        ),
    )
    wrappedIlcResult = RunFrequencyDomainIlc(
        wrappedReference,
        iqWrappedPaModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(
            numIterations=1,
            learningRate=0.15,
            regularization=1.0e-3,
            maxAmplitude=2.0,
            randomSeed=1291,
        ),
    )
    wrappedAnalysis = Analysis(
        wrappedReference,
        waveform,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 16,
            "outputFullScaleAmplitude": (
                iqWrappedPaModel.outputFullScaleAmplitude
            ),
        },
    )
    wrappedMetrics = wrappedAnalysis.Analyze(
        wrappedIlcResult.history[0].outputSignal
    )
    assert abs(wrappedMetrics["outputPowerDbm"] - 10.0) <= 0.35

    class PairedDriveOnlyPa:
        """Model a third-party PA with paired drive methods but no raw path."""

        def __init__(self) -> None:
            """Initialize one linear chain at the shared fixed-point boundary.

            Processing details:
                Algorithm: Store the public word width, physical output full
                scale, and a zero committed post-DAC drive while intentionally
                omitting raw and dual-output floating protocols.

            Returns:
                result: None. The synthetic third-party PA is ready.
            """

            self.width = 16
            self.outputFullScaleAmplitude = 2.0
            self.committedDriveDb = 0.0

        def SetCalibrationDriveDb(
            self, driveDbPerChain: Sequence[float]
        ) -> None:
            """Commit exactly one finite post-DAC drive value.

            Processing details:
                Algorithm: Validate a one-chain drive sequence and retain its
                scalar dB value for both fixed and floating public processing.

            Args:
                driveDbPerChain: Sequence containing one finite drive in dB.

            Returns:
                result: None. Later public calls use the committed drive.
            """

            driveValues = np.asarray(driveDbPerChain, dtype=float).reshape(-1)
            if driveValues.size != 1 or not np.all(np.isfinite(driveValues)):
                raise ValueError("driveDbPerChain must contain one value")
            self.committedDriveDb = float(driveValues[0])

        def ProcessCalibrationDrive(
            self,
            inputSignal: np.ndarray,
            driveDbPerChain: Sequence[float],
        ) -> np.ndarray:
            """Evaluate one uncommitted linear analog-drive trial.

            Processing details:
                Algorithm: Decode the public samples, apply the supplied
                single-chain drive once, and encode the identity-plant output
                at the declared physical output full scale.

            Args:
                inputSignal: Public fixed-point complex waveform.
                driveDbPerChain: Sequence containing one trial drive in dB.

            Returns:
                result: Public fixed-point output for the uncommitted trial.
            """

            driveValues = np.asarray(driveDbPerChain, dtype=float).reshape(-1)
            if driveValues.size != 1 or not np.all(np.isfinite(driveValues)):
                raise ValueError("driveDbPerChain must contain one value")
            assert GetFixedPointFormat(inputSignal) == (16, 1.0)
            inputFormat = FixedPoint(self.width)
            outputFormat = FixedPoint(
                self.width, self.outputFullScaleAmplitude
            )
            floatingInput = inputFormat.DecodeComplex(inputSignal)
            driveScale = np.power(10.0, float(driveValues[0]) / 20.0)
            return outputFormat.EncodeComplex(driveScale * floatingInput)

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Evaluate fixed public samples at the committed drive.

            Processing details:
                Algorithm: Reuse the uncommitted trial calculation with the
                retained drive so the physical analog gain is applied once.

            Args:
                inputSignal: Public fixed-point complex waveform.

            Returns:
                result: Public fixed-point output at the committed drive.
            """

            return self.ProcessCalibrationDrive(
                inputSignal, (self.committedDriveDb,)
            )

        def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
            """Evaluate floating public samples at the committed drive.

            Processing details:
                Algorithm: Apply the retained single-chain drive once to the
                normalized complex input without crossing a public quantizer.

            Args:
                inputSignal: Normalized complex samples before analog drive.

            Returns:
                result: Floating linear output at the committed drive.
            """

            floatingInput = np.asarray(inputSignal, dtype=np.complex128)
            driveScale = np.power(10.0, self.committedDriveDb / 20.0)
            return driveScale * floatingInput

    pairedDrivePa = PairedDriveOnlyPa()
    pairedDriveIqPa = IQImbalancePA(
        pairedDrivePa,
        directCoefficient=1.0 + 0.0j,
        imageCoefficient=0.0 + 0.0j,
    )
    pairedDriveCalibration = PowerCalibration(
        paModel=pairedDriveIqPa,
        parameters={
            "outputPowerDbm": 10.0,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 16,
        },
    )
    pairedDriveReference = pairedDriveCalibration.Calibrate(waveform.samples)
    pairedInputFormat = FixedPoint(16)
    pairedOutputFormat = FixedPoint(
        16, pairedDriveIqPa.outputFullScaleAmplitude
    )
    pairedFloatingInput = pairedInputFormat.DecodeComplex(
        pairedDriveReference
    )
    pairedFloatingOutput = pairedDriveIqPa.ProcessFloating(
        pairedFloatingInput
    )
    pairedEncodedFloatingOutput = pairedOutputFormat.EncodeComplex(
        pairedFloatingOutput
    )
    pairedFixedOutput = pairedDriveIqPa.Process(pairedDriveReference)
    assert np.array_equal(pairedEncodedFloatingOutput, pairedFixedOutput)
    pairedCalibrationDifference = (
        pairedEncodedFloatingOutput
        - pairedDriveCalibration.GetLastPaOutput()
    )
    assert np.max(np.abs(pairedCalibrationDifference.real)) <= 1.0
    assert np.max(np.abs(pairedCalibrationDifference.imag)) <= 1.0
    pairedRawOutput = pairedDriveIqPa.ProcessRawFloating(
        pairedFloatingInput
    )
    assert np.max(
        np.abs(pairedRawOutput - pairedFloatingInput)
    ) <= 2.0 / (2 ** 15)
    assert not np.allclose(pairedRawOutput, pairedFloatingOutput)
    pairedAnalysis = Analysis(
        pairedDriveReference,
        waveform,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 16,
            "outputFullScaleAmplitude": (
                pairedDriveIqPa.outputFullScaleAmplitude
            ),
        },
    )
    pairedMetrics = pairedAnalysis.Analyze(pairedFixedOutput)
    assert abs(pairedMetrics["outputPowerDbm"] - 10.0) <= 0.10

    class RawPairedDrivePa(PairedDriveOnlyPa):
        """Add an explicit raw path to the paired-drive third-party model."""

        def ProcessRawFloating(
            self, inputSignal: np.ndarray
        ) -> np.ndarray:
            """Evaluate the drive-free identity kernel.

            Processing details:
                Algorithm: Convert the supplied physical PA-input samples to
                complex128 without applying the retained committed drive.

            Args:
                inputSignal: Physical floating samples already carrying drive.

            Returns:
                result: Drive-free identity output with matching shape.
            """

            return np.asarray(inputSignal, dtype=np.complex128)

    # Wrapping an already-calibrated raw+paired third-party PA must retain the
    # inner committed drive even though the new facade has not committed its
    # own copy yet. In that state the wrapper uses the inner public path.
    preCalibratedRawPa = RawPairedDrivePa()
    preWrapperCalibration = PowerCalibration(
        paModel=preCalibratedRawPa,
        parameters={
            "outputPowerDbm": 10.0,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 16,
        },
    )
    preWrapperReference = preWrapperCalibration.Calibrate(waveform.samples)
    preWrapperInputFormat = FixedPoint(16)
    preWrapperOutputFormat = FixedPoint(
        16, preCalibratedRawPa.outputFullScaleAmplitude
    )
    preWrapperFloatingInput = preWrapperInputFormat.DecodeComplex(
        preWrapperReference
    )
    expectedPreWrapperOutput = preCalibratedRawPa.ProcessFloating(
        preWrapperFloatingInput
    )
    preCalibratedIqPa = IQImbalancePA(
        preCalibratedRawPa,
        directCoefficient=1.0 + 0.0j,
        imageCoefficient=0.0 + 0.0j,
    )
    actualPreWrapperOutput = preCalibratedIqPa.ProcessFloating(
        preWrapperFloatingInput
    )
    assert np.array_equal(
        preWrapperOutputFormat.EncodeComplex(actualPreWrapperOutput),
        preWrapperOutputFormat.EncodeComplex(expectedPreWrapperOutput),
    )
    assert np.array_equal(
        preCalibratedIqPa.Process(preWrapperReference),
        preCalibratedRawPa.Process(preWrapperReference),
    )

    thermalSampleRateHz = 100.0e3
    thermalConfig = ThermalConfig(
        enabled=True,
        modelName="single_rc",
        sampleRateHz=thermalSampleRateHz,
        thermalResistancesCPerW=(8.0,),
        thermalTimeConstantsSec=(0.01,),
        thermalUpdateIntervalSamples=20,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.01,
        maximumJunctionTemperatureC=200.0,
    )
    thermalPaModel = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 16,
        }
    )
    thermalChannel = Channel(
        paModel=thermalPaModel,
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": thermalSampleRateHz,
            "thermalRunMode": "steady_state",
            "thermalDutyCycle": 0.5,
            "thermalSteadyStateToleranceC": 1.0e-5,
            "maximumThermalSteadyStateIterations": 100,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 16,
        },
    )
    thermalSampleIndices = np.arange(200, dtype=float)
    thermalPublicInput = FixedPoint(16).EncodeComplex(
        (0.25 + 0.05j)
        * (
            1.0
            + 0.08
            * np.cos(2.0 * np.pi * thermalSampleIndices / 20.0)
        )
        * np.exp(
            1j
            * 0.12
            * np.sin(2.0 * np.pi * thermalSampleIndices / 25.0)
        )
    )
    thermalChannelOutput, thermalFeedbackOutput = thermalChannel.Process(
        thermalPublicInput,
        outputPowerDbm=20.0,
    )
    assert np.array_equal(thermalFeedbackOutput, thermalChannelOutput)
    thermalReference = thermalChannel.GetLastPaInput()
    thermalMetrics = thermalChannel.GetThermalMetrics()
    thermalPeriodDurationSec = float(
        thermalMetrics["periodDurationSec"]
    )
    assert np.isclose(
        thermalMetrics["elapsedTimeSec"],
        thermalPeriodDurationSec,
        rtol=0.0,
        atol=1.0e-15,
    )
    normalizedThermalReference = FixedPoint(16).DecodeComplex(
        thermalReference
    )
    normalizedThermalAdapter = NormalizedPaAdapter(thermalChannel)
    thermalPowerMeasurement = PowerCalibration(
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 0,
        }
    )
    for candidateScale in (0.8, 1.1):
        elapsedTimeBeforeSec = float(
            thermalChannel.GetThermalMetrics()["elapsedTimeSec"]
        )
        candidateChannelOutput, candidateFeedbackOutput = (
            normalizedThermalAdapter.ProcessOutputs(
                candidateScale * normalizedThermalReference
            )
        )
        elapsedTimeAfterSec = float(
            thermalChannel.GetThermalMetrics()["elapsedTimeSec"]
        )
        assert np.array_equal(
            candidateFeedbackOutput, candidateChannelOutput
        )
        candidateOutputRms = (
            thermalPowerMeasurement.CalculateActiveRmsPerChain(
                candidateChannelOutput
            )[0]
        )
        candidateOutputPowerDbm = (
            thermalPowerMeasurement.NormalizedRmsToOutputPowerDbm(
                candidateOutputRms
            )
        )
        assert abs(candidateOutputPowerDbm - 20.0) <= 0.10
        assert np.isclose(
            elapsedTimeAfterSec - elapsedTimeBeforeSec,
            thermalPeriodDurationSec,
            rtol=0.0,
            atol=1.0e-15,
        )

    elapsedTimeBeforeIlcSec = float(
        thermalChannel.GetThermalMetrics()["elapsedTimeSec"]
    )
    thermalIlcResult = RunFrequencyDomainIlc(
        thermalReference,
        thermalChannel,
        thermalSampleRateHz,
        20.0e3,
        ILCConfig(
            numIterations=2,
            learningRate=0.15,
            regularization=1.0e-3,
            maxAmplitude=2.0,
            randomSeed=1591,
        ),
    )
    elapsedTimeAfterIlcSec = float(
        thermalChannel.GetThermalMetrics()["elapsedTimeSec"]
    )
    # One low-power-response probe, two measured candidates, and one final
    # replay each submit exactly one live thermal period. Their calibration
    # trials are transactional and must not advance physical time.
    assert np.isclose(
        elapsedTimeAfterIlcSec - elapsedTimeBeforeIlcSec,
        4.0 * thermalPeriodDurationSec,
        rtol=0.0,
        atol=1.0e-15,
    )
    for iterationRecord in thermalIlcResult.history:
        normalizedIterationOutput = FixedPoint(
            16, thermalChannel.outputFullScaleAmplitude
        ).DecodeComplex(
            iterationRecord.outputSignal
        )
        iterationOutputRms = (
            thermalPowerMeasurement.CalculateActiveRmsPerChain(
                normalizedIterationOutput
            )[0]
        )
        iterationOutputPowerDbm = (
            thermalPowerMeasurement.NormalizedRmsToOutputPowerDbm(
                iterationOutputRms
            )
        )
        assert abs(iterationOutputPowerDbm - 20.0) <= 0.10


def CheckMainIlcPowerPointSelection() -> None:
    """Verify that the main power sweep replays the chOut-EVM best round.

    Processing details:
        Algorithm: Inject an ILC history whose feedback-domain LC-NMSE best
        round deliberately differs from its forward strict-EVM best round,
        call the main power-point helper, and require the PA replay input to
        equal the strict main-path candidate rather than ``ILCResult``'s
        native feedback-selected input.

    Returns:
        result: None. Assertions enforce strict-EVM candidate selection at the
        power-curve reporting boundary.
    """

    waveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        seed=491,
        width=0,
    ).Generate()
    pointReference = 0.20 * waveform.samples
    inputCandidates = (
        0.90 * pointReference,
        1.05 * pointReference,
        1.20 * pointReference,
    )
    channelOutputs = (
        pointReference
        + 0.08 * pointReference * np.abs(pointReference) ** 2,
        pointReference.copy(),
        pointReference + 0.12 * np.conj(pointReference),
    )
    feedbackOutputs = (
        pointReference + 0.14 * np.conj(pointReference),
        pointReference + 0.07 * np.conj(pointReference),
        pointReference.copy(),
    )
    history = [
        CalculateIterationMetrics(
            iterationIndex + 1,
            pointReference,
            feedbackOutputs[iterationIndex],
            inputCandidates[iterationIndex],
            channelOutputSignal=channelOutputs[iterationIndex],
            feedbackOutputSignal=feedbackOutputs[iterationIndex],
        )
        for iterationIndex in range(3)
    ]
    nativeBestIndex = int(
        np.argmin(
            np.asarray(
                [
                    iterationRecord.linearCompensatedNmseDb
                    for iterationRecord in history
                ],
                dtype=float,
            )
        )
    )
    assert nativeBestIndex == 2
    fakeIlcResult = ILCResult(
        learnedInput=inputCandidates[nativeBestIndex],
        outputSignal=channelOutputs[nativeBestIndex],
        history=history,
        feedbackOutputSignal=feedbackOutputs[nativeBestIndex],
    )
    paModel = PaModel(modelName="gmp", width=0)
    with patch(
        "main.RunFrequencyDomainIlc",
        return_value=fakeIlcResult,
    ), patch.object(
        paModel,
        "Process",
        return_value=channelOutputs[1],
    ) as replayMethod:
        selectedOutput = EvaluateIlcPowerPoint(
            pointReference,
            paModel,
            waveform,
            ILCConfig(numIterations=3),
            {
                "maximumOutputPowerDbm": 25.0,
                "width": 0,
            },
        )

    assert np.array_equal(selectedOutput, channelOutputs[1])
    replayMethod.assert_called_once()
    replayInput = np.asarray(
        replayMethod.call_args.args[0], dtype=np.complex128
    )
    assert np.array_equal(replayInput, inputCandidates[1])
    assert not np.array_equal(replayInput, fakeIlcResult.learnedInput)

    mimoWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        seed=492,
        numTransmitAntennas=2,
        numSpatialStreams=2,
        spatialMapping="direct",
        width=0,
    ).Generate()
    mimoReference = 0.20 * mimoWaveform.samples
    mimoInputCandidates = (
        0.90 * mimoReference,
        1.05 * mimoReference,
        1.20 * mimoReference,
    )
    mimoChannelOutputs = (
        mimoReference
        + 0.08 * mimoReference * np.abs(mimoReference) ** 2,
        mimoReference.copy(),
        mimoReference + 0.12 * np.conj(mimoReference),
    )
    mimoFeedbackOutputs = (
        mimoReference + 0.14 * np.conj(mimoReference),
        mimoReference + 0.07 * np.conj(mimoReference),
        mimoReference.copy(),
    )
    fakeChainResults = []
    for chainIndex in range(2):
        chainHistory = [
            CalculateIterationMetrics(
                iterationIndex + 1,
                mimoReference[:, chainIndex],
                mimoFeedbackOutputs[iterationIndex][:, chainIndex],
                mimoInputCandidates[iterationIndex][:, chainIndex],
                channelOutputSignal=(
                    mimoChannelOutputs[iterationIndex][:, chainIndex]
                ),
                feedbackOutputSignal=(
                    mimoFeedbackOutputs[iterationIndex][:, chainIndex]
                ),
            )
            for iterationIndex in range(3)
        ]
        fakeChainResults.append(
            ILCResult(
                learnedInput=mimoInputCandidates[2][:, chainIndex],
                outputSignal=mimoChannelOutputs[2][:, chainIndex],
                history=chainHistory,
                feedbackOutputSignal=(
                    mimoFeedbackOutputs[2][:, chainIndex]
                ),
            )
        )
    fakeMimoIlcResult = MimoIlcResult(
        learnedInput=mimoInputCandidates[2],
        outputSignal=mimoChannelOutputs[2],
        chainResults=tuple(fakeChainResults),
        feedbackOutputSignal=mimoFeedbackOutputs[2],
    )
    mimoPaModel = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "width": 0,
        }
    )
    with patch(
        "main.RunMimoFrequencyDomainIlc",
        return_value=fakeMimoIlcResult,
    ), patch.object(
        mimoPaModel,
        "Process",
        return_value=mimoChannelOutputs[1],
    ) as mimoReplayMethod:
        selectedMimoOutput = EvaluateIlcPowerPoint(
            mimoReference,
            mimoPaModel,
            mimoWaveform,
            ILCConfig(numIterations=3),
            {
                "maximumOutputPowerDbm": 25.0,
                "width": 0,
            },
        )

    assert np.array_equal(selectedMimoOutput, mimoChannelOutputs[1])
    mimoReplayMethod.assert_called_once()
    mimoReplayInput = np.asarray(
        mimoReplayMethod.call_args.args[0], dtype=np.complex128
    )
    assert np.array_equal(mimoReplayInput, mimoInputCandidates[1])
    assert not np.array_equal(
        mimoReplayInput, fakeMimoIlcResult.learnedInput
    )


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
        exercise mixed Doherty/GMP MIMO configuration, require the smoother
        built-in Doherty defaults to remain nonfolding, and reject invalid
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

    defaultDohertyConfig = DohertyConfig()
    assert defaultDohertyConfig.peakingInputGain == 0.85
    assert defaultDohertyConfig.defaultWienerSaturationAmplitude == 2.0
    assert defaultDohertyConfig.peakingTransitionWidth == 0.50
    assert defaultDohertyConfig.peakingCombineCoefficient == 0.045 + 0.0j
    assert defaultDohertyConfig.loadModulationStrength == 0.0035
    defaultDoherty = DohertyPA(defaultDohertyConfig)
    assert defaultDoherty.carrierModel.config.saturationAmplitude == 2.0
    assert defaultDoherty.peakingModel.config.saturationAmplitude == 2.0
    directHelperBranch = DohertyPA.BuildBranchModel(
        "wiener",
        None,
        None,
    )
    assert directHelperBranch.config.saturationAmplitude == 1.55
    defaultAmplitudeGrid = np.linspace(0.0, 2.0, 401)
    defaultSettledMagnitude = np.asarray(
        tuple(
            abs(
                defaultDoherty.Process(
                    np.full(16, amplitudeValue, dtype=np.complex128)
                )[-1]
            )
            for amplitudeValue in defaultAmplitudeGrid
        )
    )
    assert np.all(np.diff(defaultSettledMagnitude) >= -1.0e-10)

    invalidDohertyConfigs = (
        {"carrierModelName": "memoryPolynomial"},
        {"peakingInputGain": 0.0},
        {"defaultWienerSaturationAmplitude": 0.0},
        {"defaultWienerSaturationAmplitude": float("nan")},
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


def CheckRappPaModel() -> None:
    """Verify the classic Rapp PA is nonlinear but strictly memoryless.

    Processing details:
        Algorithm: Compare the implementation with the closed-form AM-AM
        equation, prove that phase and sample order are preserved, show that
        surrounding samples cannot change a selected output, exercise the
        facade and fixed-point boundary, and reject every nonpositive or
        nonfinite physical parameter.

    Returns:
        result: None. Assertions enforce the memoryless Rapp contract.
    """

    rappConfig = RappConfig(
        linearGain=1.2,
        saturationAmplitude=0.9,
        rappSmoothness=2.5,
    )
    rappPa = RappPA(rappConfig)
    inputMagnitudes = np.asarray(
        (0.0, 0.05, 0.25, 0.75, 1.5), dtype=float
    )
    inputPhases = np.asarray(
        (0.0, 0.2, -0.7, 1.1, -2.0), dtype=float
    )
    inputSignal = inputMagnitudes * np.exp(1j * inputPhases)
    outputSignal = rappPa.Process(inputSignal)
    expectedMagnitude = (
        rappConfig.linearGain
        * inputMagnitudes
        / (
            1.0
            + (
                inputMagnitudes / rappConfig.saturationAmplitude
            )
            ** (2.0 * rappConfig.rappSmoothness)
        )
        ** (1.0 / (2.0 * rappConfig.rappSmoothness))
    )
    assert np.allclose(np.abs(outputSignal), expectedMagnitude)
    nonzeroMask = inputMagnitudes > 0.0
    assert np.allclose(
        np.angle(outputSignal[nonzeroMask]),
        inputPhases[nonzeroMask],
    )
    assert np.isclose(rappPa.SmallSignalGain(), 1.2 + 0.0j)

    # Replacing every neighboring sample must not alter the same-index output.
    probeIndex = 2
    changedContext = np.asarray(inputSignal, dtype=np.complex128).copy()
    changedContext[:probeIndex] = 0.8 - 0.3j
    changedContext[probeIndex + 1:] = -0.4 + 0.6j
    changedOutput = rappPa.Process(changedContext)
    assert changedOutput[probeIndex] == outputSignal[probeIndex]

    facadePa = PaModel(
        parameters={
            "modelName": "rapp",
            "rappConfig": rappConfig,
            "width": 0,
        }
    )
    assert facadePa.modelName == "rapp"
    assert np.allclose(facadePa.Process(inputSignal), outputSignal)
    fixedFormat = FixedPoint(16)
    fixedInput = fixedFormat.EncodeComplex(0.5 * inputSignal)
    fixedPa = PaModel(
        parameters={
            "modelName": "rapp",
            "rappConfig": rappConfig,
            "width": 16,
        }
    )
    fixedOutput = fixedPa.Process(fixedInput)
    assert np.array_equal(fixedOutput.real, np.rint(fixedOutput.real))
    assert np.array_equal(fixedOutput.imag, np.rint(fixedOutput.imag))

    invalidConfigurations = (
        {"linearGain": 0.0},
        {"saturationAmplitude": 0.0},
        {"rappSmoothness": 0.0},
        {"linearGain": float("nan")},
    )
    for invalidOverrides in invalidConfigurations:
        try:
            RappPA(RappConfig(**invalidOverrides))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                "invalid Rapp configuration accepted: "
                f"{invalidOverrides!r}"
            )


def CheckGmpPaModel() -> None:
    """Verify physically consistent default GMP steady and transient behavior.

    Processing details:
        Algorithm: Sum every same-order default basis to confirm the 13.5%
        nonlinear strength and that memory depth cannot move the settled
        AM-AM curve, verify the normalized long cubic envelope-memory branch
        has zero settled coefficient sum, excite the default PA with a
        continuous high-envelope plateau to bound startup droop, and prove
        that explicitly supplied measured coefficients retain exact behavior.

    Returns:
        result: None. Assertions enforce the default GMP coefficient contract.
    """

    referenceSteadyCoefficients = {
        1: 1.261692 + 0.014052j,
        3: -0.291144 + 0.054204j,
        5: 0.031812 - 0.022452j,
        7: -0.000168 + 0.002784j,
    }
    expectedSteadyCoefficients = {
        nonlinearOrder: coefficient
        * (1.0 if nonlinearOrder == 1 else 0.135)
        for nonlinearOrder, coefficient in (
            referenceSteadyCoefficients.items()
        )
    }
    for memoryDepth in (1, 3, 5):
        for crossMemoryDepth in (0, 2, 4):
            (
                mainCoefficients,
                laggingCoefficients,
                leadingCoefficients,
            ) = DefaultGmpCoefficients(
                tuple(expectedSteadyCoefficients),
                memoryDepth,
                crossMemoryDepth,
            )
            for nonlinearOrder, expectedCoefficient in (
                expectedSteadyCoefficients.items()
            ):
                combinedCoefficient = sum(
                    coefficient
                    for (order, _), coefficient in mainCoefficients.items()
                    if order == nonlinearOrder
                ) + sum(
                    coefficient
                    for (order, _, _), coefficient in (
                        laggingCoefficients.items()
                    )
                    if order == nonlinearOrder
                ) + sum(
                    coefficient
                    for (order, _, _), coefficient in (
                        leadingCoefficients.items()
                    )
                    if order == nonlinearOrder
                )
                assert np.isclose(
                    combinedCoefficient,
                    expectedCoefficient,
                    rtol=0.0,
                    atol=1e-12,
                )

    # Use a nondefault explicit value to verify exact coefficient accounting;
    # the configured built-in default is asserted separately above.
    explicitLongMemoryCoefficient = 0.008 - 0.0036j
    (
        longMainCoefficients,
        longLaggingCoefficients,
        longLeadingCoefficients,
    ) = DefaultGmpCoefficients(
        tuple(expectedSteadyCoefficients),
        3,
        2,
        longEnvelopeMemoryDepth=12,
        longEnvelopeMemoryDecay=0.82,
        longEnvelopeMemoryCoefficient=explicitLongMemoryCoefficient,
    )
    assert max(
        crossIndex
        for order, _memoryIndex, crossIndex in longLaggingCoefficients
        if order == 3
    ) == 12
    ordinaryMain, ordinaryLagging, _ = DefaultGmpCoefficients(
        tuple(expectedSteadyCoefficients),
        3,
        2,
    )
    addedLongLaggingCoefficient = sum(
        coefficient
        - ordinaryLagging.get(coefficientKey, 0.0 + 0.0j)
        for coefficientKey, coefficient in longLaggingCoefficients.items()
        if coefficientKey[0] == 3
    )
    assert np.isclose(
        addedLongLaggingCoefficient,
        explicitLongMemoryCoefficient,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.isclose(
        longMainCoefficients[(3, 0)]
        - ordinaryMain[(3, 0)],
        -explicitLongMemoryCoefficient,
        rtol=0.0,
        atol=1.0e-15,
    )
    longSteadyCubicCoefficient = sum(
        coefficient
        for (order, _), coefficient in longMainCoefficients.items()
        if order == 3
    ) + sum(
        coefficient
        for (order, _, _), coefficient in longLaggingCoefficients.items()
        if order == 3
    ) + sum(
        coefficient
        for (order, _, _), coefficient in longLeadingCoefficients.items()
        if order == 3
    )
    assert np.isclose(
        longSteadyCubicCoefficient,
        expectedSteadyCoefficients[3],
        rtol=0.0,
        atol=1.0e-12,
    )
    defaultGeneratedGmp = GMPPA()
    assert max(
        crossIndex
        for order, _memoryIndex, crossIndex in (
            defaultGeneratedGmp.laggingCoefficients
        )
        if order == 3
    ) == 12
    disabledLongMemoryGmp = GMPPA(
        GMPConfig(longEnvelopeMemoryDepth=0)
    )
    assert max(
        crossIndex
        for order, _memoryIndex, crossIndex in (
            disabledLongMemoryGmp.laggingCoefficients
        )
        if order == 3
    ) == 2

    # The full fitted reference remains available explicitly for stress
    # plants such as the built-in piecewise GMP model, while the ordinary
    # GMP default is intentionally gentler at 20 dBm.
    (
        referenceMainCoefficients,
        referenceLaggingCoefficients,
        referenceLeadingCoefficients,
    ) = DefaultGmpCoefficients(
        tuple(referenceSteadyCoefficients),
        3,
        2,
        nonlinearScale=1.0,
    )
    for nonlinearOrder, expectedCoefficient in (
        referenceSteadyCoefficients.items()
    ):
        combinedReferenceCoefficient = sum(
            coefficient
            for (order, _), coefficient in (
                referenceMainCoefficients.items()
            )
            if order == nonlinearOrder
        ) + sum(
            coefficient
            for (order, _, _), coefficient in (
                referenceLaggingCoefficients.items()
            )
            if order == nonlinearOrder
        ) + sum(
            coefficient
            for (order, _, _), coefficient in (
                referenceLeadingCoefficients.items()
            )
            if order == nonlinearOrder
        )
        assert np.isclose(
            combinedReferenceCoefficient,
            expectedCoefficient,
            rtol=0.0,
            atol=1.0e-12,
        )

    for invalidNonlinearScale in (-0.01, 1.01, float("nan")):
        try:
            GMPConfig(nonlinearScale=invalidNonlinearScale).Validate()
        except ValueError as error:
            assert "nonlinearScale" in str(error)
        else:
            raise AssertionError("invalid GMP nonlinearScale accepted")
    invalidLongMemoryConfigurations = (
        {"longEnvelopeMemoryDepth": -1},
        {"longEnvelopeMemoryDepth": 1.5},
        {"longEnvelopeMemoryDecay": 0.0},
        {"longEnvelopeMemoryDecay": 1.01},
        {"longEnvelopeMemoryCoefficient": complex(float("nan"), 0.0)},
    )
    for invalidOverrides in invalidLongMemoryConfigurations:
        try:
            GMPConfig(**invalidOverrides).Validate()
        except ValueError as error:
            assert "longEnvelopeMemory" in str(error)
        else:
            raise AssertionError(
                "invalid GMP long-memory configuration accepted: "
                f"{invalidOverrides!r}"
            )

    # The fitted default static curve must remain nondecreasing throughout its
    # documented normalized input interval instead of folding back and forcing
    # power calibration onto a remote polynomial expansion branch.
    amplitudeGrid = np.linspace(0.0, 2.0, 2001)
    steadyOutputMagnitude = np.abs(
        sum(
            coefficient * amplitudeGrid**nonlinearOrder
            for nonlinearOrder, coefficient in (
                expectedSteadyCoefficients.items()
            )
        )
    )
    assert np.all(np.diff(steadyOutputMagnitude) >= -1e-12)

    # The default generator also accepts reduced or extended odd-order sets.
    # Every automatically generated set must remain monotonic in the same
    # declared interval; unknown higher orders default to zero rather than
    # creating an uncontrolled polynomial extrapolation term.
    for defaultOrderSet in (
        (1,),
        (1, 3),
        (1, 5),
        (1, 3, 5),
        (1, 3, 7),
        (1, 5, 7),
        (1, 3, 5, 7),
        (7, 5, 3, 1),
        (1, 3, 3, 5, 7),
        (1, 3, 5, 7, 9),
        (3, 5, 7),
    ):
        (
            orderMainCoefficients,
            orderLaggingCoefficients,
            orderLeadingCoefficients,
        ) = DefaultGmpCoefficients(defaultOrderSet, 3, 2)
        orderSteadyCoefficients = {
            nonlinearOrder: sum(
                coefficient
                for (order, _), coefficient in (
                    orderMainCoefficients.items()
                )
                if order == nonlinearOrder
            ) + sum(
                coefficient
                for (order, _, _), coefficient in (
                    orderLaggingCoefficients.items()
                )
                if order == nonlinearOrder
            ) + sum(
                coefficient
                for (order, _, _), coefficient in (
                    orderLeadingCoefficients.items()
                )
                if order == nonlinearOrder
            )
            for nonlinearOrder in defaultOrderSet
        }
        orderOutputMagnitude = np.abs(
            sum(
                coefficient * amplitudeGrid**nonlinearOrder
                for nonlinearOrder, coefficient in (
                    orderSteadyCoefficients.items()
                )
            )
        )
        assert np.all(np.diff(orderOutputMagnitude) >= -1e-10)
        if 9 in defaultOrderSet:
            assert np.isclose(orderSteadyCoefficients[9], 0.0 + 0.0j)

    # A high-level run preceded by zeros exposes excessive repeated nonlinear
    # memory coefficients. The phase-dominant long-memory residual must keep
    # every plateau amplitude within a small ripple while still settling
    # exactly instead of producing an ongoing high-level collapse.
    plateauPhaseRadians = 0.37
    plateauLength = 40
    for plateauAmplitude in (0.25, 0.5, 0.9, 1.2, 1.5, 1.7, 2.0):
        plateauInput = np.concatenate(
            (
                np.zeros(8, dtype=np.complex128),
                np.full(
                    plateauLength,
                    plateauAmplitude * np.exp(1j * plateauPhaseRadians),
                    dtype=np.complex128,
                ),
                np.zeros(8, dtype=np.complex128),
            )
        )
        plateauOutput = GMPPA().Process(plateauInput)
        plateauMagnitude = np.abs(plateauOutput[8:8 + plateauLength])
        settledMagnitude = float(np.mean(plateauMagnitude[-8:]))
        onsetToSettledDb = 20.0 * np.log10(
            plateauMagnitude[0] / settledMagnitude
        )
        plateauRippleDb = 20.0 * np.log10(
            np.max(plateauMagnitude) / np.min(plateauMagnitude)
        )
        assert abs(onsetToSettledDb) <= 0.45
        assert plateauRippleDb <= 0.45
        assert np.allclose(
            plateauMagnitude[-8:],
            settledMagnitude,
            rtol=0.0,
            atol=1.0e-12,
        )

    # Default stabilization must never override coefficients extracted from a
    # measured PA.  This deliberately strong two-tap model retains its exact
    # sample-to-sample response because all dictionaries are explicit.
    measuredGmp = GMPPA(
        GMPConfig(
            nonlinearOrders=(1,),
            memoryDepth=2,
            crossMemoryDepth=0,
            mainCoefficients={(1, 0): 1.0 + 0.0j, (1, 1): -0.5 + 0.0j},
            laggingCoefficients={},
            leadingCoefficients={},
        )
    )
    measuredInput = np.ones(4, dtype=np.complex128)
    assert np.allclose(
        measuredGmp.Process(measuredInput),
        np.asarray((1.0, 0.5, 0.5, 0.5), dtype=np.complex128),
    )
    measuredNonlinearGmp = GMPPA(
        GMPConfig(
            nonlinearOrders=(1, 3),
            memoryDepth=1,
            crossMemoryDepth=0,
            mainCoefficients={
                (1, 0): 1.0 + 0.0j,
                (3, 0): 0.1 - 0.02j,
            },
        )
    )
    nonlinearMeasuredInput = np.asarray(
        (0.2 + 0.1j, 0.7 - 0.3j, -0.4 + 0.8j),
        dtype=np.complex128,
    )
    assert np.allclose(
        measuredNonlinearGmp.Process(nonlinearMeasuredInput),
        nonlinearMeasuredInput
        + (0.1 - 0.02j)
        * nonlinearMeasuredInput
        * np.abs(nonlinearMeasuredInput) ** 2,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert measuredNonlinearGmp.laggingCoefficients == {}


def CheckPiecewiseGmpModels() -> None:
    """Verify smooth piecewise PA and DPD models on independent records.

    Processing details:
        Algorithm: Check custom and default PA regions, C2 partition weights,
        identity inference, joint piecewise regression against a global GMP,
        and noisy-data validation with and without adjacent-region coefficient
        smoothing. The validation waveform is independent of all fit samples.

    Returns:
        result: None. Assertions enforce continuity, monotonic PA response,
            regional model accuracy, and effective smoothness regularization.
    """

    pureGainConfigs = tuple(
        GMPConfig(
            nonlinearOrders=(1,),
            memoryDepth=1,
            crossMemoryDepth=0,
            mainCoefficients={(1, 0): gainValue},
            laggingCoefficients={},
            leadingCoefficients={},
        )
        for gainValue in (
            1.10 + 0.00j,
            0.95 + 0.02j,
            0.80 - 0.03j,
        )
    )
    customPiecewiseConfig = PiecewiseGMPConfig(
        regionBoundaries=(0.30, 0.70),
        transitionWidths=(0.10, 0.10),
        regionConfigs=pureGainConfigs,
    )
    customPiecewisePa = PiecewiseGMPPA(customPiecewiseConfig)
    regionalInput = np.asarray(
        (0.10 + 0.0j, 0.40 + 0.0j, 0.90 + 0.0j),
        dtype=np.complex128,
    )
    expectedRegionalOutput = regionalInput * np.asarray(
        tuple(
            regionConfig.mainCoefficients[(1, 0)]
            for regionConfig in pureGainConfigs
        )
    )
    assert np.allclose(
        customPiecewisePa.Process(regionalInput),
        expectedRegionalOutput,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert customPiecewisePa.SmallSignalGain() == 1.10 + 0.00j

    generatedRegionalPa = PiecewiseGMPPA(
        PiecewiseGMPConfig(
            regionConfigs=(GMPConfig(), GMPConfig(), GMPConfig()),
        )
    )
    assert all(
        max(
            crossIndex
            for order, _memoryIndex, crossIndex in (
                regionModel.laggingCoefficients
            )
            if order == 3
        )
        == 12
        for regionModel in generatedRegionalPa.regionModels
    )

    defaultPiecewisePa = PiecewiseGMPPA()
    assert all(
        max(
            crossIndex
            for order, _memoryIndex, crossIndex in (
                regionModel.laggingCoefficients
            )
            if order == 3
        )
        == 12
        for regionModel in defaultPiecewisePa.regionModels
    )
    amplitudeGrid = np.linspace(0.0, 2.0, 401)
    settledMagnitudes = np.asarray(
        tuple(
            abs(
                defaultPiecewisePa.Process(
                    np.full(32, amplitudeValue, dtype=np.complex128)
                )[-1]
            )
            for amplitudeValue in amplitudeGrid
        )
    )
    assert np.all(np.diff(settledMagnitudes) >= -1.0e-10)
    facadePa = PaModel(
        parameters={
            "modelName": "piecewise_gmp",
            "piecewiseGmpConfig": customPiecewiseConfig,
            "width": 0,
        }
    )
    assert isinstance(facadePa.model, PiecewiseGMPPA)
    assert facadePa.model.config == customPiecewiseConfig
    assert np.allclose(
        facadePa.Process(regionalInput),
        customPiecewisePa.Process(regionalInput),
        rtol=0.0,
        atol=1.0e-14,
    )
    try:
        PiecewiseGMPConfig(
            regionBoundaries=(0.70, 0.30),
            transitionWidths=(0.10, 0.10),
            regionConfigs=pureGainConfigs,
        ).Validate()
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("unordered piecewise-GMP boundaries accepted")

    commonDpdParameters = {
        "nonlinearOrders": (1, 3),
        "memoryDepth": 1,
        "crossMemoryDepth": 0,
        "ridgeFactor": 1.0e-8,
        "maximumOutputMagnitude": None,
        "width": 0,
    }
    identityPiecewiseDpd = PiecewiseDpdGmp(
        parameters=commonDpdParameters
    )
    weightProbe = np.asarray(
        (0.05, 0.25, 0.40, 0.60, 0.90),
        dtype=np.complex128,
    )
    envelopeWeights = identityPiecewiseDpd.CalculateEnvelopeWeights(
        weightProbe
    )
    assert envelopeWeights.shape == (weightProbe.size, 3)
    assert np.all(envelopeWeights >= 0.0)
    assert np.all(envelopeWeights <= 1.0)
    assert np.allclose(
        np.sum(envelopeWeights, axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.allclose(
        identityPiecewiseDpd.Process(weightProbe),
        weightProbe,
        rtol=0.0,
        atol=2.0e-15,
    )

    randomGenerator = np.random.default_rng(20260828)

    def BuildEnvelopeRecord(sampleCount: int) -> np.ndarray:
        """Create one independent full-region complex-envelope record.

        Processing details:
            Algorithm: Draw amplitudes uniformly across all three regions and
            phases uniformly around the complex plane using the enclosing
            test's deterministic random generator.

        Args:
            sampleCount: Positive number of requested complex samples.

        Returns:
            result: Complex vector spanning low, middle, and high envelopes.
        """

        amplitudes = randomGenerator.uniform(0.02, 1.0, sampleCount)
        phases = randomGenerator.uniform(-np.pi, np.pi, sampleCount)
        return amplitudes * np.exp(1j * phases)

    trainingReference = BuildEnvelopeRecord(5000)
    validationReference = BuildEnvelopeRecord(5000)
    teacherDpd = PiecewiseDpdGmp(
        parameters={
            **commonDpdParameters,
            "regionSmoothnessFactor": 0.0,
        }
    )
    teacherDpd.SetCoefficients(
        np.asarray(
            (
                1.02 + 0.00j,
                0.06 - 0.01j,
                1.00 + 0.01j,
                0.12 - 0.02j,
                0.96 + 0.03j,
                0.22 - 0.04j,
            ),
            dtype=np.complex128,
        )
    )
    trainingTarget = teacherDpd.Process(trainingReference)
    validationTarget = teacherDpd.Process(validationReference)
    globalDpd = DpdGmp(parameters=commonDpdParameters)
    globalDpd.Fit(trainingReference, trainingTarget)
    fittedPiecewiseDpd = PiecewiseDpdGmp(
        parameters={
            **commonDpdParameters,
            "regionSmoothnessFactor": 0.0,
        }
    )
    fittedPiecewiseDpd.Fit(trainingReference, trainingTarget)
    globalValidationNmseDb = globalDpd.CalculateNmse(
        validationReference,
        validationTarget,
    )
    piecewiseValidationNmseDb = fittedPiecewiseDpd.CalculateNmse(
        validationReference,
        validationTarget,
    )
    assert piecewiseValidationNmseDb < globalValidationNmseDb - 50.0
    assert np.allclose(
        fittedPiecewiseDpd.GetRegionCoefficients("middle"),
        teacherDpd.GetRegionCoefficients("middle"),
        rtol=0.0,
        atol=2.0e-5,
    )

    noisyDpdParameters = {
        **commonDpdParameters,
        "nonlinearOrders": (1, 3, 5),
    }
    noisyTeacher = PiecewiseDpdGmp(
        parameters={
            **noisyDpdParameters,
            "regionSmoothnessFactor": 0.0,
        }
    )
    noisyTeacher.SetCoefficients(
        np.asarray(
            (
                1.02 + 0.00j,
                0.06 - 0.01j,
                0.005 + 0.000j,
                1.00 + 0.01j,
                0.12 - 0.02j,
                0.008 + 0.000j,
                0.96 + 0.03j,
                0.22 - 0.04j,
                0.012 + 0.000j,
            ),
            dtype=np.complex128,
        )
    )
    noisyTrainingReference = BuildEnvelopeRecord(800)
    noisyValidationReference = BuildEnvelopeRecord(8000)
    cleanTrainingTarget = noisyTeacher.Process(noisyTrainingReference)
    noisyTrainingTarget = cleanTrainingTarget + 0.004 * (
        randomGenerator.normal(size=noisyTrainingReference.size)
        + 1j
        * randomGenerator.normal(size=noisyTrainingReference.size)
    )
    noisyValidationTarget = noisyTeacher.Process(
        noisyValidationReference
    )
    unsmoothedDpd = PiecewiseDpdGmp(
        parameters={
            **noisyDpdParameters,
            "regionSmoothnessFactor": 0.0,
        }
    )
    smoothedDpd = PiecewiseDpdGmp(
        parameters={
            **noisyDpdParameters,
            "regionSmoothnessFactor": 0.1,
        }
    )
    unsmoothedResult = unsmoothedDpd.Fit(
        noisyTrainingReference,
        noisyTrainingTarget,
    )
    smoothedResult = smoothedDpd.Fit(
        noisyTrainingReference,
        noisyTrainingTarget,
    )
    unsmoothedValidationNmseDb = unsmoothedDpd.CalculateNmse(
        noisyValidationReference,
        noisyValidationTarget,
    )
    smoothedValidationNmseDb = smoothedDpd.CalculateNmse(
        noisyValidationReference,
        noisyValidationTarget,
    )
    assert (
        smoothedResult.regionSmoothnessPenalty
        < 0.01 * unsmoothedResult.regionSmoothnessPenalty
    )
    assert smoothedValidationNmseDb < unsmoothedValidationNmseDb - 0.5
    assert (
        smoothedResult.afterNmseDb
        < unsmoothedResult.afterNmseDb + 0.1
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
        assert iterationRecord.outputSignal.size == (
            referenceSignal.size + delaySamples
        )
        assert iterationRecord.feedbackOutputSignal is not None
        assert iterationRecord.feedbackOutputSignal.size == (
            referenceSignal.size + delaySamples
        )
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
    """Verify tolerant classes warn while Channel rejects unknown names.

    Processing details:
        Algorithm: Exercise constructor mappings, direct keyword overrides,
        live external edits, and update methods across tolerant ChainMap-backed
        classes; then verify Channel fails fast for every top-level and nested
        unknown-name entry path.

    Returns:
        result: None. Assertions enforce each class's documented policy.
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

    strictConstructorCases = (
        (
            "constructor mapping",
            lambda: Channel(
                parameters={
                    "width": 0,
                    "txiqgainimbalancedb": 0.5,
                }
            ),
            "txiqgainimbalancedb",
        ),
        (
            "constructor keyword",
            lambda: Channel(
                width=0,
                unknownChannelKeyword=1,
            ),
            "unknownChannelKeyword",
        ),
        (
            "nested coupling path",
            lambda: Channel(
                parameters={
                    "prePaCouplingPaths": (
                        {
                            "sourceChain": 0,
                            "destinationChain": 1,
                            "unknownPathSetting": 1,
                        },
                    ),
                }
            ),
            "unknownPathSetting",
        ),
    )
    strictErrorTexts = {}
    for caseName, strictConstructor, expectedUnknownName in (
        strictConstructorCases
    ):
        try:
            strictConstructor()
        except TypeError as error:
            errorText = str(error)
            strictErrorTexts[caseName] = errorText
            assert expectedUnknownName in errorText
            assert "case-sensitive" in errorText
            assert "highest to lowest similarity" in errorText
        else:
            raise AssertionError(
                f"Channel {caseName} must reject unknown names"
            )

    validChannelParameterNames = tuple(
        Channel(parameters={"width": 0}).GetParameters()
    )
    topLevelRankingText = strictErrorTexts[
        "constructor mapping"
    ].split("txiqgainimbalancedb: ", 1)[1].splitlines()[0]
    rankedTopLevelNames = tuple(topLevelRankingText.split(", "))
    assert rankedTopLevelNames[0] == "txIqGainImbalanceDb"
    assert len(rankedTopLevelNames) == len(validChannelParameterNames)
    assert set(rankedTopLevelNames) == set(validChannelParameterNames)

    rankedPathNames = tuple(
        strictErrorTexts["nested coupling path"]
        .split("unknownPathSetting: ", 1)[1]
        .splitlines()[0]
        .split(", ")
    )
    assert set(rankedPathNames) == {
        "sourceChain",
        "destinationChain",
        "gainDb",
        "phaseDegrees",
        "integerDelaySamples",
        "fractionalDelaySamples",
        "firTaps",
    }

    strictUpdateChannel = Channel(parameters={"width": 0})
    try:
        strictUpdateChannel.UpdateParameters(
            unknownChannelUpdate=2
        )
    except TypeError as error:
        assert "unknownChannelUpdate" in str(error)
    else:
        raise AssertionError(
            "Channel.UpdateParameters must reject unknown names"
        )

    liveChannelParameters = {"width": 0}
    liveStrictChannel = Channel(parameters=liveChannelParameters)
    liveChannelParameters["lateUnknownChannelSetting"] = 3
    try:
        liveStrictChannel.GetParameters()
    except TypeError as error:
        assert "lateUnknownChannelSetting" in str(error)
    else:
        raise AssertionError(
            "Channel must reject unknown names added to a live mapping"
        )

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
        require a complex128 metadata-bearing public container in both modes,
        exercise 14- and 16-bit generators, and pass raw codes through PA and
        Analysis.

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
    assert isinstance(floatingSignal, FixedPointArray)
    assert isinstance(fixedSignal, FixedPointArray)
    assert GetFixedPointFormat(floatingSignal) == (0, 1.0)
    assert GetFixedPointFormat(fixedSignal) == (3, 1.0)
    assert GetFixedPointFormat(fixedSignal.copy()) == (3, 1.0)
    assert GetFixedPointFormat(fixedSignal[1:]) == (3, 1.0)
    assert floatingSignal.shape == fixedSignal.shape == inputSignal.shape
    assert np.array_equal(floatingSignal, inputSignal)
    assert np.array_equal(fixedSignal, expectedFixedSignal)
    decodedFixedSignal = fixedFormat.DecodeComplex(fixedSignal)
    assert np.array_equal(decodedFixedSignal, expectedDecodedSignal)
    assert type(decodedFixedSignal) is np.ndarray
    assert GetFixedPointFormat(decodedFixedSignal) is None
    assert np.array_equal(
        fixedFormat.QuantizeCodes(
            np.array([1.4 - 5.2j, -9.0 + 2.6j])
        ),
        np.array([1.0 - 4.0j, -4.0 + 3.0j]),
    )
    assert fixedFormat.GetFormatInfo()["minimumCode"] == -4.0
    assert fixedFormat.GetFormatInfo()["maximumCode"] == 3.0

    physicalFormat = FixedPoint(
        width=3,
        fullScaleAmplitude=2.0,
    )
    physicalSignal = np.array(
        [0.5 + 1.0j, 2.5 - 3.0j],
        dtype=np.complex128,
    )
    physicalCodes = physicalFormat.EncodeComplex(physicalSignal)
    assert GetFixedPointFormat(physicalCodes) == (3, 2.0)
    assert np.array_equal(
        physicalCodes,
        np.array([1.0 + 2.0j, 3.0 - 4.0j]),
    )
    assert np.array_equal(
        physicalFormat.DecodeComplex(physicalCodes),
        np.array([0.5 + 1.0j, 1.5 - 2.0j]),
    )
    physicalFormatInfo = physicalFormat.GetFormatInfo()
    assert physicalFormatInfo["fullScaleAmplitude"] == 2.0
    assert physicalFormatInfo["quantizationStep"] == 0.5
    assert physicalFormatInfo["physicalMinimumValue"] == -2.0
    assert physicalFormatInfo["physicalMaximumValue"] == 1.5

    class FixedOnlyOutputScalePa:
        """Expose an identity plant only through asymmetric fixed formats."""

        width = 16
        outputFullScaleAmplitude = 2.0

        def Process(self, publicInput: np.ndarray) -> np.ndarray:
            """Pass one physical signal through asymmetric public scales.

            Processing details:
                Algorithm: Decode normalized 16-bit DAC codes, preserve the
                physical complex samples exactly, and re-encode them using
                the synthetic PA's two-times-larger output full scale.

            Args:
                publicInput: Public Q1 fixed-point I/Q code vector.

            Returns:
                result: Equal physical samples encoded as Q2 output codes.
            """

            floatingInput = FixedPoint(self.width).DecodeComplex(publicInput)
            return FixedPoint(
                self.width, self.outputFullScaleAmplitude
            ).EncodeComplex(floatingInput)

    fixedOnlyProbe = np.array(
        [0.5 + 0.25j, -0.375 + 0.125j],
        dtype=np.complex128,
    )
    fixedOnlyWrapper = IQImbalancePA(
        FixedOnlyOutputScalePa(),
        directCoefficient=1.0 + 0.0j,
        imageCoefficient=0.0 + 0.0j,
    )
    assert np.allclose(
        fixedOnlyWrapper.ProcessFloating(fixedOnlyProbe),
        FixedPoint(16, 2.0).DecodeComplex(
            FixedPoint(16, 2.0).EncodeComplex(fixedOnlyProbe)
        ),
        rtol=0.0,
        atol=1.0e-12,
    )

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

    for invalidFullScale in (
        0.0,
        -1.0,
        float("inf"),
        float("nan"),
        True,
        "2.0",
    ):
        try:
            FixedPoint(16, invalidFullScale)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                "invalid fixed-point full scale accepted: "
                f"{invalidFullScale!r}"
            )

    publicScaleConstructors = (
        (
            "PaModel",
            lambda value: PaModel(outputFullScaleAmplitude=value),
        ),
        (
            "MimoPaModel",
            lambda value: MimoPaModel(
                outputFullScaleAmplitude=value
            ),
        ),
        (
            "Channel",
            lambda value: Channel(outputFullScaleAmplitude=value),
        ),
        (
            "Analysis",
            lambda value: Analysis(
                defaultWaveform.samples,
                defaultWaveform,
                outputFullScaleAmplitude=value,
            ),
        ),
    )
    for invalidFullScale in (True, np.bool_(True), "2.0"):
        for constructorName, constructor in publicScaleConstructors:
            try:
                constructor(invalidFullScale)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(
                    f"{constructorName} accepted invalid output full scale "
                    f"{invalidFullScale!r}"
                )

        invalidThirdPartyPa = FixedOnlyOutputScalePa()
        invalidThirdPartyPa.outputFullScaleAmplitude = invalidFullScale
        invalidScaleChannel = Channel(
            paModel=invalidThirdPartyPa,
            width=16,
        )
        invalidScaleWrapper = IQImbalancePA(invalidThirdPartyPa)
        thirdPartyScaleOperations = (
            (
                "Channel.ProcessPaOutput",
                lambda: invalidScaleChannel.ProcessPaOutput(
                    np.zeros(1, dtype=np.complex128)
                ),
            ),
            (
                "Channel.ProcessBoundPaFloating",
                lambda: invalidScaleChannel.ProcessBoundPaFloating(
                    np.zeros(1, dtype=np.complex128)
                ),
            ),
            (
                "IQImbalancePA.outputFullScaleAmplitude",
                lambda: invalidScaleWrapper.outputFullScaleAmplitude,
            ),
        )
        for operationName, operation in thirdPartyScaleOperations:
            try:
                operation()
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(
                    f"{operationName} accepted third-party output full scale "
                    f"{invalidFullScale!r}"
                )


def CheckPaThermalModel() -> None:
    """Verify thermal power, duty cycle, drift, idle cooling, and calibration.

    Processing details:
        Algorithm: Compare static and Foster state behavior, confirm higher
        duty cycle produces more heat, prove reference-temperature calibration
        does not advance thermal time, and explicitly select transient Channel
        scheduling before verifying that cold calibration precedes each live
        thermal transmission while output power remains free to drift.

    Returns:
        result: None. Assertions identify thermal state or workflow failures.
    """

    recommendedStatic = ThermalConfig.Recommended(
        "static",
        sampleRateHz=100.0e3,
    )
    recommendedSingleRc = ThermalConfig.Recommended(
        "single_rc",
        sampleRateHz=100.0e3,
    )
    recommendedFoster = ThermalConfig.Recommended(
        "foster",
        sampleRateHz=100.0e3,
        ambientTemperatureC=30.0,
        initialJunctionTemperatureC=30.0,
    )
    assert recommendedStatic.enabled
    assert recommendedStatic.modelName == "static"
    assert recommendedStatic.initialJunctionTemperatureC == 55.0
    assert recommendedStatic.thermalResistancesCPerW == (1.0,)
    assert recommendedStatic.thermalTimeConstantsSec == (1.0,)
    assert recommendedSingleRc.thermalResistancesCPerW == (20.0,)
    assert recommendedSingleRc.thermalTimeConstantsSec == (20.0e-3,)
    assert recommendedFoster.thermalResistancesCPerW == (
        2.0,
        8.0,
        20.0,
    )
    assert recommendedFoster.thermalTimeConstantsSec == (
        50.0e-6,
        5.0e-3,
        0.5,
    )
    assert recommendedFoster.ambientTemperatureC == 30.0
    assert recommendedFoster.initialJunctionTemperatureC == 30.0
    for invalidModelName in ("single", "cauer"):
        try:
            ThermalConfig.Recommended(invalidModelName)
        except ValueError as error:
            assert "'static', 'single_rc', or 'foster'" in str(error)
        else:
            raise AssertionError(
                f"invalid recommended thermal model accepted: "
                f"{invalidModelName}"
            )
    try:
        ThermalConfig.Recommended(
            "foster",
            unknownThermalParameter=1.0,
        )
    except TypeError as error:
        assert "Supported parameters" in str(error)
    else:
        raise AssertionError("unknown recommended thermal parameter accepted")

    thermalConfig = ThermalConfig(
        enabled=True,
        modelName="single_rc",
        sampleRateHz=100.0e3,
        thermalResistancesCPerW=(20.0,),
        thermalTimeConstantsSec=(0.02,),
        thermalUpdateIntervalSamples=100,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.08,
        maximumJunctionTemperatureC=150.0,
    )
    thermalNetwork = ThermalNetwork(thermalConfig)
    assert np.isclose(thermalNetwork.CurrentTemperatureC(), 25.0)
    heatedTemperature = thermalNetwork.Advance(1.0, 0.02)
    assert heatedTemperature > 25.0
    cooledTemperature = thermalNetwork.Advance(0.0, 0.02)
    assert 25.0 < cooledTemperature < heatedTemperature
    staticConfig = replace(
        thermalConfig,
        modelName="static",
        initialJunctionTemperatureC=55.0,
    )
    staticNetwork = ThermalNetwork(staticConfig)
    staticNetwork.Advance(10.0, 1.0)
    assert np.isclose(staticNetwork.CurrentTemperatureC(), 55.0)
    fosterConfig = replace(
        thermalConfig,
        modelName="foster",
        thermalResistancesCPerW=(2.0, 8.0, 20.0),
        thermalTimeConstantsSec=(50.0e-6, 5.0e-3, 0.5),
    )
    fosterNetwork = ThermalNetwork(fosterConfig)
    fosterNetwork.Advance(1.0, 0.01)
    assert 25.0 < fosterNetwork.CurrentTemperatureC() < 55.0

    for modelName in ("rapp", "wiener", "gmp", "doherty"):
        wrappedPa = PaModel(
            parameters={
                "modelName": modelName,
                "thermalConfig": thermalConfig,
                "width": 0,
            }
        )
        wrappedOutput = wrappedPa.Process(
            np.full(500, 0.25 + 0.05j)
        )
        assert np.all(np.isfinite(wrappedOutput))
        assert wrappedPa.GetThermalMetrics()[
            "endingJunctionTemperatureC"
        ] > 25.0

    continuousPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    burstPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    continuousInput = np.full(2000, 0.45 + 0.0j)
    burstInput = continuousInput.copy()
    burstInput[1000:] = 0.0
    continuousPa.Process(continuousInput)
    burstPa.Process(burstInput)
    continuousMetrics = continuousPa.GetThermalMetrics()
    burstMetrics = burstPa.GetThermalMetrics()
    assert np.isclose(continuousMetrics["activeSampleDutyCycle"], 1.0)
    assert np.isclose(burstMetrics["activeSampleDutyCycle"], 0.5)
    assert (
        continuousMetrics["endingJunctionTemperatureC"]
        > burstMetrics["endingJunctionTemperatureC"]
    )

    thermalPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    thermalChannel = Channel(
        paModel=thermalPa,
        parameters={
            "sampleRateHz": thermalConfig.sampleRateHz,
            "thermalRunMode": "transient",
            "thermalDutyCycle": 1.0,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )
    rawInput = np.full(2000, 0.35 + 0.0j)
    firstOutput, firstFeedbackOutput = thermalChannel.Process(
        rawInput,
        outputPowerDbm=20.0,
    )
    assert firstFeedbackOutput.shape == firstOutput.shape
    firstRms = float(np.sqrt(np.mean(np.abs(firstOutput) ** 2)))
    firstMetrics = thermalChannel.GetThermalMetrics()
    firstCalibrationMetrics = (
        thermalChannel.GetLastCalibrationMetrics()
    )
    assert firstCalibrationMetrics["converged"]
    assert abs(
        firstCalibrationMetrics[
            "measuredOutputPowerDbmPerChain"
        ][0]
        - 20.0
    ) <= 0.05
    assert (
        firstMetrics["outputPowerDbm"]
        < firstCalibrationMetrics[
            "measuredOutputPowerDbmPerChain"
        ][0]
    )
    assert not np.allclose(
        firstOutput,
        thermalChannel.GetLastPaOutput(),
    )
    assert np.isclose(firstMetrics["elapsedTimeSec"], 0.02)
    assert firstMetrics["endingJunctionTemperatureC"] > 25.0

    # Supplying the same reference target again must not close a loop around
    # the current hot output. Calibration is repeated with temperature effects
    # suspended, the hot snapshot is restored, and only the real frame advances
    # thermal time. Consequently the fixed reference drive still exhibits
    # temperature-dependent output drift.
    secondOutput, secondFeedbackOutput = thermalChannel.Process(
        rawInput,
        outputPowerDbm=20.0,
    )
    assert secondFeedbackOutput.shape == secondOutput.shape
    secondRms = float(np.sqrt(np.mean(np.abs(secondOutput) ** 2)))
    secondMetrics = thermalChannel.GetThermalMetrics()
    assert secondMetrics["endingJunctionTemperatureC"] > firstMetrics[
        "endingJunctionTemperatureC"
    ]
    assert secondRms < firstRms
    assert secondMetrics["elapsedTimeSec"] > firstMetrics["elapsedTimeSec"]
    assert np.isclose(secondMetrics["elapsedTimeSec"], 0.04)
    temperatureBeforeIdle = secondMetrics["endingJunctionTemperatureC"]
    thermalChannel.AdvanceThermalIdle(0.05)
    assert (
        thermalChannel.GetThermalMetrics()["junctionTemperatureC"]
        < temperatureBeforeIdle
    )

    mimoThermalPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {"modelName": "wiener", "thermalConfig": thermalConfig},
                {"modelName": "wiener", "thermalConfig": thermalConfig},
            ),
            "thermalCouplingCPerW": (
                (0.0, 2.0),
                (3.0, 0.0),
            ),
            "width": 0,
        }
    )
    mimoThermalInput = np.column_stack(
        (continuousInput, 0.2 * continuousInput)
    )
    mimoThermalPa.Process(mimoThermalInput)
    mimoMetrics = mimoThermalPa.GetThermalMetrics()["chains"]
    assert mimoMetrics[0]["averageDissipatedPowerW"] > mimoMetrics[1][
        "averageDissipatedPowerW"
    ]
    assert mimoThermalPa.paModels[1]._externalTemperatureOffsetC > 0.0


def CheckThermalDisableAndCalibrationBypass() -> None:
    """Verify authoritative thermal disable and cold calibration transactions.

    Processing details:
        Algorithm: Compare disabled and absent thermal models at floating and
        fixed public boundaries, reject direct construction of a disabled
        ThermalNetwork, clear a retained MIMO heating offset on a live disable,
        exercise explicit suspension and live disable before restoration, and
        prove successful and failed direct calibrations preserve exact thermal
        state while observing the reference-temperature electrical PA. Repeat
        the transaction through a bound Process method, an IQ wrapper, and a
        small MIMO bank, then verify a thermally disabled steady-mode Channel
        never requires an output-power target.

    Returns:
        result: None. Assertions identify thermal-disable or calibration-state
        regressions at every supported public PA boundary.
    """

    disabledThermalConfig = ThermalConfig(
        enabled=False,
        modelName="single_rc",
        sampleRateHz=100.0e3,
        ambientTemperatureC=25.0,
        initialJunctionTemperatureC=125.0,
        referenceTemperatureC=25.0,
        thermalResistancesCPerW=(40.0,),
        thermalTimeConstantsSec=(0.01,),
        thermalUpdateIntervalSamples=16,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.10,
        phaseTemperatureCoefficientDegreesPerC=0.25,
        saturationTemperatureCoefficientPerC=-0.003,
        nonlinearityTemperatureCoefficientPerC=0.01,
        maximumJunctionTemperatureC=150.0,
    )
    enabledThermalConfig = replace(
        disabledThermalConfig,
        enabled=True,
        initialJunctionTemperatureC=70.0,
    )
    floatingInput = 0.23 * np.exp(
        1j * np.linspace(0.0, 3.0 * np.pi, 192, endpoint=False)
    )

    # A disabled ThermalConfig is a PaModel switch, not a request to build a
    # lower-level network that could accidentally ignore the switch.
    try:
        ThermalNetwork(disabledThermalConfig)
    except ValueError as error:
        assert "enabled=True" in str(error)
    else:
        raise AssertionError(
            "ThermalNetwork accepted ThermalConfig(enabled=False)"
        )

    # Disabled thermal parameters, even deliberately extreme ones, must be
    # numerically indistinguishable from omitting thermalConfig altogether.
    for interfaceWidth in (0, 16):
        noThermalPa = PaModel(
            parameters={"modelName": "rapp", "width": interfaceWidth}
        )
        disabledThermalPa = PaModel(
            parameters={
                "modelName": "rapp",
                "thermalConfig": disabledThermalConfig,
                "width": interfaceWidth,
            }
        )
        publicInput = (
            floatingInput
            if interfaceWidth == 0
            else FixedPoint(interfaceWidth).EncodeComplex(floatingInput)
        )
        referenceOutput = noThermalPa.Process(publicInput)
        disabledOutput = disabledThermalPa.Process(publicInput)
        assert np.array_equal(disabledOutput, referenceOutput)
        assert disabledThermalPa.GetThermalMetrics() == {"enabled": False}
        assert disabledThermalPa.thermalNetwork is None
        assert np.isclose(
            disabledThermalPa._externalTemperatureOffsetC, 0.0
        )

    # A live True-to-False update must erase neighbor-induced temperature
    # state immediately and must not retain a hidden offset for re-enablement.
    liveDisablePa = PaModel(
        parameters={
            "modelName": "rapp",
            "thermalConfig": enabledThermalConfig,
            "width": 0,
        }
    )
    liveDisablePa.SetExternalTemperatureOffsetC(9.0)
    assert np.isclose(liveDisablePa._externalTemperatureOffsetC, 9.0)
    liveDisablePa.UpdateParameters(
        thermalConfig=disabledThermalConfig
    )
    assert liveDisablePa.GetThermalMetrics() == {"enabled": False}
    assert liveDisablePa.thermalNetwork is None
    assert np.isclose(liveDisablePa._externalTemperatureOffsetC, 0.0)
    assert np.array_equal(
        liveDisablePa.Process(floatingInput),
        PaModel(modelName="rapp", width=0).Process(floatingInput),
    )
    liveDisablePa.SetExternalTemperatureOffsetC(12.0)
    assert np.isclose(liveDisablePa._externalTemperatureOffsetC, 0.0)

    # Suspension must bypass every temperature-application entry, not only
    # the usual Process path. Nested transactions are rejected because one
    # snapshot cannot safely represent two independently owned restores.
    transactionPa = PaModel(
        parameters={
            "modelName": "rapp",
            "thermalConfig": enabledThermalConfig,
            "width": 0,
        }
    )
    coldElectricalOutput = PaModel(
        modelName="rapp", width=0
    ).ProcessFloating(floatingInput)
    hotElectricalOutput = transactionPa.ProcessAtTemperatureFloating(
        floatingInput, 125.0
    )
    assert not np.allclose(hotElectricalOutput, coldElectricalOutput)
    transactionPa.SetExternalTemperatureOffsetC(6.0)
    thermalSnapshot = transactionPa.SuspendThermalModel()
    assert thermalSnapshot is not None
    assert np.array_equal(
        transactionPa.ApplyTemperatureDrift(
            coldElectricalOutput, 125.0
        ),
        coldElectricalOutput,
    )
    assert np.array_equal(
        transactionPa.ProcessAtTemperatureFloating(
            floatingInput, 125.0
        ),
        coldElectricalOutput,
    )
    try:
        transactionPa.SuspendThermalModel()
    except RuntimeError as error:
        assert "already suspended" in str(error)
    else:
        raise AssertionError("nested thermal suspension was accepted")

    # A live disable issued during the suspended interval is authoritative;
    # restoring the older snapshot must not revive heat or mutual coupling.
    transactionPa.UpdateParameters(
        thermalConfig=disabledThermalConfig
    )
    transactionPa.RestoreThermalModel(thermalSnapshot)
    assert transactionPa.GetThermalMetrics() == {"enabled": False}
    assert transactionPa.thermalNetwork is None
    assert np.isclose(transactionPa._externalTemperatureOffsetC, 0.0)
    assert np.array_equal(
        transactionPa.Process(floatingInput), coldElectricalOutput
    )

    # Warm the PA and add a mutual-heating contribution before calibration so
    # exact restoration covers branch temperatures, elapsed time, diagnostics,
    # and external temperature rise rather than only a pristine network.
    hotCalibrationPa = PaModel(
        parameters={
            "modelName": "rapp",
            "thermalConfig": enabledThermalConfig,
            "width": 0,
        }
    )
    warmupSignal = np.full(
        160, 0.42 + 0.03j, dtype=np.complex128
    )
    hotCalibrationPa.Process(warmupSignal)
    hotCalibrationPa.SetExternalTemperatureOffsetC(4.0)
    thermalMetricsBeforeSuccess = hotCalibrationPa.GetThermalMetrics()
    assert thermalMetricsBeforeSuccess["elapsedTimeSec"] > 0.0
    assert np.isclose(
        thermalMetricsBeforeSuccess["mutualHeatingTemperatureRiseC"],
        4.0,
    )
    calibrationInput = np.concatenate(
        (
            np.zeros(16, dtype=np.complex128),
            floatingInput,
            np.zeros(16, dtype=np.complex128),
        )
    )
    commonCalibrationParameters = {
        "outputPowerDbm": 18.0,
        "maximumOutputPowerDbm": 25.0,
        "calibrationToleranceDb": 0.02,
        "maximumCalibrationIterations": 30,
        "width": 0,
    }
    hotCalibration = PowerCalibration(
        paModel=hotCalibrationPa,
        parameters=commonCalibrationParameters,
    )
    # The electrical loop is deliberately exposed only as a guarded kernel.
    # Bypassing Calibrate must fail before either PA or thermal state changes.
    thermalMetricsBeforeRejectedKernel = (
        hotCalibrationPa.GetThermalMetrics()
    )
    try:
        hotCalibration.CalibrateElectricalOnly(calibrationInput)
    except RuntimeError as error:
        assert "internal numerical kernel" in str(error)
        assert "call Calibrate" in str(error)
    else:
        raise AssertionError(
            "CalibrateElectricalOnly ran outside its thermal transaction"
        )
    assert (
        hotCalibrationPa.GetThermalMetrics()
        == thermalMetricsBeforeRejectedKernel
    )
    coldCalibrationPa = PaModel(modelName="rapp", width=0)
    coldCalibration = PowerCalibration(
        paModel=coldCalibrationPa,
        parameters=commonCalibrationParameters,
    )
    hotCalibratedInput = hotCalibration.Calibrate(calibrationInput)
    coldCalibratedInput = coldCalibration.Calibrate(calibrationInput)
    assert np.array_equal(hotCalibratedInput, coldCalibratedInput)
    assert np.array_equal(
        hotCalibration.GetLastPaOutput(),
        coldCalibration.GetLastPaOutput(),
    )
    assert (
        hotCalibration.GetLastCalibrationMetrics()
        == coldCalibration.GetLastCalibrationMetrics()
    )
    assert (
        hotCalibrationPa.GetThermalMetrics()
        == thermalMetricsBeforeSuccess
    )
    assert np.isclose(
        hotCalibrationPa._externalTemperatureOffsetC, 4.0
    )

    # Passing PaModel.Process directly must discover the owning PA's width,
    # drive protocol, and thermal transaction through the bound method.
    thermalMetricsBeforeBoundMethod = (
        hotCalibrationPa.GetThermalMetrics()
    )
    boundMethodCalibration = PowerCalibration(
        paModel=hotCalibrationPa.Process,
        parameters=commonCalibrationParameters,
    )
    boundMethodInput = boundMethodCalibration.Calibrate(calibrationInput)
    assert np.array_equal(boundMethodInput, coldCalibratedInput)
    assert np.array_equal(
        boundMethodCalibration.GetLastPaOutput(),
        coldCalibration.GetLastPaOutput(),
    )
    assert (
        hotCalibrationPa.GetThermalMetrics()
        == thermalMetricsBeforeBoundMethod
    )

    # Even an exception from a deliberately under-budgeted calibration must
    # execute the finally restore and leave the exact hot snapshot untouched.
    failureParameters = {
        "outputPowerDbm": 24.37,
        "calibrationToleranceDb": 1.0e-12,
        "maximumCalibrationIterations": 1,
    }
    hotCalibration.UpdateParameters(**failureParameters)
    coldCalibration.UpdateParameters(**failureParameters)
    thermalMetricsBeforeFailure = hotCalibrationPa.GetThermalMetrics()
    try:
        hotCalibration.Calibrate(calibrationInput)
    except RuntimeError as error:
        assert "did not converge" in str(error)
    else:
        raise AssertionError(
            "one-iteration hot calibration unexpectedly converged"
        )
    try:
        coldCalibration.Calibrate(calibrationInput)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "one-iteration cold calibration unexpectedly converged"
        )
    assert (
        hotCalibrationPa.GetThermalMetrics()
        == thermalMetricsBeforeFailure
    )
    hotFailureMetrics = hotCalibration.GetLastCalibrationMetrics()
    coldFailureMetrics = coldCalibration.GetLastCalibrationMetrics()
    assert hotFailureMetrics["converged"] is False
    assert coldFailureMetrics["converged"] is False
    assert np.allclose(
        hotFailureMetrics["measuredOutputPowerDbmPerChain"],
        coldFailureMetrics["measuredOutputPowerDbmPerChain"],
        rtol=0.0,
        atol=1.0e-12,
    )

    # A hostile measurement callback must not replace the PA after its old
    # thermal snapshot has been captured. Otherwise the finalizer could restore
    # that snapshot into the wrong physical device and strand the old PA in its
    # suspended state.
    class RebindingThermalPlant:
        """Attempt to replace the bound PA from inside a calibration trial."""

        def __init__(
            self,
            wrappedPa: PaModel,
            replacementPa: PaModel,
        ) -> None:
            """Retain two thermal PAs and an initially unbound calibrator.

            Processing details:
                Algorithm: Expose the wrapped PA's public width and paired
                thermal protocol while retaining a distinct replacement PA
                that the malicious Process callback will try to install.

            Args:
                wrappedPa: Original PA whose thermal state will be suspended.
                replacementPa: Different PA proposed during the active trial.

            Returns:
                result: None. The adversarial callback is ready for binding.
            """

            self.wrappedPa = wrappedPa
            self.replacementPa = replacementPa
            self.calibrator = None
            self.width = wrappedPa.width

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Try to rebind the calibrator before evaluating the old PA.

            Processing details:
                Algorithm: Require the test to install its PowerCalibration,
                request an illegal active-transaction PA replacement, and
                delegate only if that safety guard unexpectedly permits it.

            Args:
                inputSignal: Floating calibration trial waveform.

            Returns:
                result: Wrapped-PA output only if the forbidden rebind succeeds.
            """

            if not isinstance(self.calibrator, PowerCalibration):
                raise RuntimeError("test calibrator was not attached")
            self.calibrator.SetPaModel(self.replacementPa)
            return self.wrappedPa.Process(inputSignal)

        def SuspendThermalModel(self) -> object:
            """Snapshot and suspend the original wrapped thermal PA.

            Processing details:
                Algorithm: Delegate snapshot ownership to the original PA so
                the test can verify that calibration restores the same owner.

            Returns:
                result: Opaque snapshot created by the original wrapped PA.
            """

            return self.wrappedPa.SuspendThermalModel()

        def RestoreThermalModel(self, thermalSnapshot: object) -> None:
            """Restore a captured snapshot only to the original wrapped PA.

            Processing details:
                Algorithm: Delegate the final transaction step to the PA that
                created the snapshot, never to the proposed replacement PA.

            Args:
                thermalSnapshot: Opaque original-PA thermal snapshot.

            Returns:
                result: None. The original PA resumes its prior hot state.
            """

            self.wrappedPa.RestoreThermalModel(thermalSnapshot)

    replacementPa = PaModel(
        parameters={
            "modelName": "rapp",
            "thermalConfig": enabledThermalConfig,
            "width": 0,
        }
    )
    replacementPa.Process(0.5 * warmupSignal)
    replacementPa.SetExternalTemperatureOffsetC(2.0)
    originalMetricsBeforeRebind = hotCalibrationPa.GetThermalMetrics()
    replacementMetricsBeforeRebind = replacementPa.GetThermalMetrics()
    rebindingPlant = RebindingThermalPlant(
        hotCalibrationPa, replacementPa
    )
    rebindingCalibration = PowerCalibration(
        paModel=rebindingPlant,
        parameters=commonCalibrationParameters,
    )
    rebindingPlant.calibrator = rebindingCalibration
    try:
        rebindingCalibration.Calibrate(calibrationInput[:128])
    except RuntimeError as error:
        assert "cannot rebind" in str(error)
        assert "active calibration transaction" in str(error)
    else:
        raise AssertionError(
            "calibration callback replaced its PA during a transaction"
        )
    assert (
        hotCalibrationPa.GetThermalMetrics()
        == originalMetricsBeforeRebind
    )
    assert (
        replacementPa.GetThermalMetrics()
        == replacementMetricsBeforeRebind
    )
    assert hotCalibrationPa._thermalEffectsSuspended is False
    assert replacementPa._thermalEffectsSuspended is False

    # IQ and MIMO facades must forward the same transaction protocol. Small
    # waveforms and relaxed tolerances keep this architectural check quick.
    iqThermalPa = PaModel(
        parameters={
            "modelName": "rapp",
            "thermalConfig": enabledThermalConfig,
            "width": 0,
        }
    )
    iqThermalPa.Process(warmupSignal)
    iqWrappedPa = IQImbalancePA(
        iqThermalPa,
        directCoefficient=0.99 + 0.01j,
        imageCoefficient=0.02 - 0.005j,
    )
    iqMetricsBeforeCalibration = iqWrappedPa.GetThermalMetrics()
    iqCalibration = PowerCalibration(
        paModel=iqWrappedPa,
        parameters={
            "outputPowerDbm": 16.0,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "maximumCalibrationIterations": 24,
            "width": 0,
        },
    )
    iqCalibration.Calibrate(calibrationInput[:128])
    assert iqWrappedPa.GetThermalMetrics() == iqMetricsBeforeCalibration

    mimoCalibrationPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {
                    "modelName": "rapp",
                    "thermalConfig": enabledThermalConfig,
                },
                {
                    "modelName": "rapp",
                    "thermalConfig": enabledThermalConfig,
                },
            ),
            "thermalCouplingCPerW": (
                (0.0, 1.0),
                (1.5, 0.0),
            ),
            "width": 0,
        }
    )
    mimoInput = np.column_stack(
        (calibrationInput[:128], 0.7j * calibrationInput[:128])
    )
    mimoCalibrationPa.Process(mimoInput)
    mimoMetricsBeforeCalibration = (
        mimoCalibrationPa.GetThermalMetrics()
    )
    mimoCalibration = PowerCalibration(
        paModel=mimoCalibrationPa,
        parameters={
            "outputPowerDbmPerChain": (16.0, 17.0),
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "maximumCalibrationIterations": 24,
            "width": 0,
        },
    )
    mimoCalibration.Calibrate(mimoInput)
    assert (
        mimoCalibrationPa.GetThermalMetrics()
        == mimoMetricsBeforeCalibration
    )

    # Default steady-state scheduling is irrelevant when the PA reports that
    # thermal effects are disabled, so a first target-free call remains valid.
    disabledChannel = Channel(
        paModel=PaModel(
            parameters={
                "modelName": "rapp",
                "thermalConfig": disabledThermalConfig,
                "width": 0,
            }
        ),
        parameters={"width": 0},
    )
    targetFreeOutput, targetFreeFeedbackOutput = disabledChannel.Process(
        floatingInput
    )
    assert targetFreeOutput.shape == floatingInput.shape
    assert targetFreeFeedbackOutput.shape == floatingInput.shape
    assert np.all(np.isfinite(targetFreeOutput))
    assert np.all(np.isfinite(targetFreeFeedbackOutput))
    assert disabledChannel.GetThermalMetrics() == {"enabled": False}


def CheckChannelPeriodicThermalModes() -> None:
    """Verify Channel periodic steady-state and transient thermal scheduling.

    Processing details:
        Algorithm: Validate every scheduling control, use a waveform containing
        an internal idle region to distinguish configured and actual duty,
        require a closed steady-state temperature cycle, prove every steady
        call repeats reference-temperature power calibration, compare causal
        transient periods, and exercise per-chain MIMO duty reporting.

    Returns:
        result: None. Assertions identify periodic thermal regressions.
    """

    defaultParameters = Channel(parameters={"width": 0}).GetParameters()
    assert defaultParameters["thermalRunMode"] == "steady_state"
    assert np.isclose(defaultParameters["thermalDutyCycle"], 1.0)
    assert defaultParameters["thermalSteadyStateToleranceC"] > 0.0
    assert defaultParameters["maximumThermalSteadyStateIterations"] >= 1

    invalidThermalParameters = (
        ({"thermalRunMode": "periodic"}, "thermalRunMode", "Allowed values"),
        ({"thermalRunMode": 1}, "thermalRunMode", "Allowed values"),
        ({"thermalDutyCycle": 0.0}, "thermalDutyCycle", "Allowed range"),
        ({"thermalDutyCycle": 1.01}, "thermalDutyCycle", "Allowed range"),
        ({"thermalDutyCycle": True}, "thermalDutyCycle", "Allowed range"),
        (
            {"thermalSteadyStateToleranceC": 0.0},
            "thermalSteadyStateToleranceC",
            "Allowed range",
        ),
        (
            {"thermalSteadyStateToleranceC": float("nan")},
            "thermalSteadyStateToleranceC",
            "Allowed range",
        ),
        (
            {"maximumThermalSteadyStateIterations": 0},
            "maximumThermalSteadyStateIterations",
            "Allowed range",
        ),
        (
            {"maximumThermalSteadyStateIterations": 1.5},
            "maximumThermalSteadyStateIterations",
            "Allowed range",
        ),
    )
    for invalidParameters, parameterName, expectedPhrase in (
        invalidThermalParameters
    ):
        try:
            Channel(parameters={"width": 0, **invalidParameters})
        except ValueError as error:
            errorMessage = str(error)
            assert parameterName in errorMessage
            assert expectedPhrase in errorMessage
        else:
            raise AssertionError(
                f"invalid periodic thermal setting accepted: {parameterName}"
            )

    thermalConfig = ThermalConfig(
        enabled=True,
        modelName="single_rc",
        sampleRateHz=100.0e3,
        thermalResistancesCPerW=(20.0,),
        thermalTimeConstantsSec=(5.0e-3,),
        thermalUpdateIntervalSamples=50,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.03,
        maximumJunctionTemperatureC=200.0,
    )
    activeSamples = np.full(500, 0.25 + 0.05j, dtype=np.complex128)
    idleSamples = np.zeros(500, dtype=np.complex128)
    periodicInput = np.concatenate((activeSamples, idleSamples))
    steadyPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    steadyChannel = Channel(
        paModel=steadyPa,
        parameters={
            "sampleRateHz": thermalConfig.sampleRateHz,
            "thermalRunMode": "steady_state",
            "thermalDutyCycle": 0.4,
            "thermalSteadyStateToleranceC": 1.0e-6,
            "maximumThermalSteadyStateIterations": 100,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )

    # The configured duty counts the complete data window. Half of this test
    # window is internally idle, so its true RF-active duty is 0.4 * 0.5.
    assert np.isclose(steadyChannel.GetActualDutyCycle(periodicInput), 0.2)
    try:
        steadyChannel.Process(periodicInput)
    except ValueError as error:
        assert "requires outputPowerDbm" in str(error)
    else:
        raise AssertionError(
            "the first steady-state period ran without a power target"
        )

    firstSteadyOutput, firstSteadyFeedbackOutput = steadyChannel.Process(
        periodicInput,
        outputPowerDbm=18.0,
    )
    assert firstSteadyFeedbackOutput.shape == firstSteadyOutput.shape
    firstSteadyMetrics = steadyChannel.GetThermalMetrics()
    firstCalibrationMetrics = steadyChannel.GetLastCalibrationMetrics()
    assert firstCalibrationMetrics["converged"]
    assert firstCalibrationMetrics["targetOutputPowerDbmPerChain"] == (18.0,)
    assert abs(
        firstCalibrationMetrics["measuredOutputPowerDbmPerChain"][0] - 18.0
    ) <= 0.05
    assert firstSteadyMetrics["thermalRunMode"] == "steady_state"
    assert firstSteadyMetrics["steadyStateConverged"]
    assert firstSteadyMetrics["steadyStateIterations"] >= 1
    assert firstSteadyMetrics["steadyStateErrorC"] <= 1.0e-6
    assert np.isclose(firstSteadyMetrics["configuredDutyCycle"], 0.4)
    assert np.isclose(firstSteadyMetrics["waveformActiveDutyCycle"], 0.5)
    assert np.isclose(firstSteadyMetrics["activeSampleDutyCycle"], 0.5)
    assert np.isclose(firstSteadyMetrics["actualDutyCycle"], 0.2)
    assert np.isclose(steadyChannel.GetActualDutyCycle(), 0.2)
    assert np.isclose(firstSteadyMetrics["signalDurationSec"], 0.01)
    assert np.isclose(firstSteadyMetrics["periodDurationSec"], 0.025)
    assert np.isclose(
        firstSteadyMetrics["scheduledIdleDurationSec"], 0.015
    )
    assert np.isclose(firstSteadyMetrics["elapsedTimeSec"], 0.025)
    assert abs(
        firstSteadyMetrics["periodEndingJunctionTemperatureC"]
        - firstSteadyMetrics["periodStartingJunctionTemperatureC"]
    ) <= 1.0e-6
    assert (
        firstSteadyMetrics["dataEndingJunctionTemperatureC"]
        > firstSteadyMetrics["periodStartingJunctionTemperatureC"]
    )
    temperatureTrace = np.asarray(
        firstSteadyMetrics["temperatureTraceC"], dtype=float
    )
    activityTrace = tuple(
        firstSteadyMetrics["temperatureTraceRfActive"]
    )
    assert any(activityTrace)
    assert not all(activityTrace)
    assert np.max(temperatureTrace) > temperatureTrace[-2]
    assert temperatureTrace[-1] < temperatureTrace[-2]

    # Omitting the target after the first accepted request must reuse 18 dBm
    # and recalibrate. Scaling the raw active samples by five therefore leaves
    # the accepted steady-state output unchanged rather than reducing it by 5x.
    secondSteadyOutput, secondSteadyFeedbackOutput = steadyChannel.Process(
        0.2 * periodicInput
    )
    assert secondSteadyFeedbackOutput.shape == secondSteadyOutput.shape
    secondSteadyMetrics = steadyChannel.GetThermalMetrics()
    secondCalibrationMetrics = steadyChannel.GetLastCalibrationMetrics()
    assert secondCalibrationMetrics["converged"]
    assert secondCalibrationMetrics["targetOutputPowerDbmPerChain"] == (18.0,)
    assert abs(
        secondCalibrationMetrics["measuredOutputPowerDbmPerChain"][0] - 18.0
    ) <= 0.05
    assert np.allclose(
        secondSteadyOutput,
        firstSteadyOutput,
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    assert np.isclose(secondSteadyMetrics["elapsedTimeSec"], 0.05)
    assert abs(
        secondSteadyMetrics["periodEndingJunctionTemperatureC"]
        - secondSteadyMetrics["periodStartingJunctionTemperatureC"]
    ) <= 1.0e-6

    transientPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": thermalConfig,
            "width": 0,
        }
    )
    transientChannel = Channel(
        paModel=transientPa,
        parameters={
            "sampleRateHz": thermalConfig.sampleRateHz,
            "thermalRunMode": "transient",
            "thermalDutyCycle": 0.4,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )
    transientChannel.Process(periodicInput, outputPowerDbm=18.0)
    firstTransientMetrics = transientChannel.GetThermalMetrics()
    transientChannel.Process(periodicInput, outputPowerDbm=18.0)
    secondTransientMetrics = transientChannel.GetThermalMetrics()
    assert firstTransientMetrics["thermalRunMode"] == "transient"
    assert (
        firstTransientMetrics["periodEndingJunctionTemperatureC"]
        > firstTransientMetrics["periodStartingJunctionTemperatureC"]
    )
    assert np.isclose(
        secondTransientMetrics["periodStartingJunctionTemperatureC"],
        firstTransientMetrics["periodEndingJunctionTemperatureC"],
    )
    assert (
        secondTransientMetrics["periodEndingJunctionTemperatureC"]
        > firstTransientMetrics["periodEndingJunctionTemperatureC"]
    )
    assert np.isclose(secondTransientMetrics["elapsedTimeSec"], 0.05)

    mimoThermalPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {"modelName": "wiener", "thermalConfig": thermalConfig},
                {"modelName": "wiener", "thermalConfig": thermalConfig},
            ),
            "thermalCouplingCPerW": (
                (0.0, 1.0),
                (2.0, 0.0),
            ),
            "width": 0,
        }
    )
    mimoChannel = Channel(
        paModel=mimoThermalPa,
        parameters={
            "sampleRateHz": thermalConfig.sampleRateHz,
            "thermalRunMode": "steady_state",
            "thermalDutyCycle": 0.4,
            "thermalSteadyStateToleranceC": 1.0e-5,
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )
    secondChain = np.concatenate(
        (
            np.full(250, 0.25 + 0.05j, dtype=np.complex128),
            np.zeros(750, dtype=np.complex128),
        )
    )
    mimoInput = np.column_stack((periodicInput, secondChain))
    assert np.allclose(mimoChannel.GetActualDutyCycle(mimoInput), (0.2, 0.1))
    mimoOutput, mimoFeedbackOutput = mimoChannel.Process(
        mimoInput,
        outputPowerDbm=(16.0, 16.0),
    )
    assert mimoOutput.shape == mimoInput.shape
    assert mimoFeedbackOutput.shape == mimoInput.shape
    assert np.allclose(mimoChannel.GetActualDutyCycle(), (0.2, 0.1))
    completeMimoMetrics = mimoChannel.GetThermalMetrics()
    mimoMetrics = completeMimoMetrics["chains"]
    assert len(mimoMetrics) == 2
    assert completeMimoMetrics["mutualHeating"]["steadyStateConverged"]
    assert (
        completeMimoMetrics["mutualHeating"]["steadyStateErrorC"]
        <= 1.0e-5
    )
    assert all(
        chainMetrics["steadyStateConverged"] for chainMetrics in mimoMetrics
    )
    assert all(
        abs(
            chainMetrics["periodEndingJunctionTemperatureC"]
            - chainMetrics["periodStartingJunctionTemperatureC"]
        )
        <= 1.0e-5
        for chainMetrics in mimoMetrics
    )


def CheckPeriodicThermalEdgeCases() -> None:
    """Verify periodic-thermal reference planes and atomic transactions.

    Processing details:
        Algorithm: Measure actual RF duty without a ThermalConfig, reject each
        Channel/PA time-power-activity reference-plane mismatch before power
        calibration can commit state, force the second chain of an uncoupled
        MIMO PA to fail after the first chain advances and require a complete
        rollback, then prove an IQ-imbalance wrapper preserves steady-state
        scheduling, outer idle time, and per-call power recalibration.

    Returns:
        result: None. Assertions identify periodic thermal edge regressions.
    """

    activeSamples = np.full(
        500, 0.25 + 0.05j, dtype=np.complex128
    )
    idleSamples = np.zeros(500, dtype=np.complex128)
    periodicInput = np.concatenate((activeSamples, idleSamples))

    # Actual-duty observation remains useful before self-heating is enabled.
    # Channel classifies activity at its PA-input reference plane, so a 50%
    # active data window scheduled for 40% of the period has 20% RF duty.
    nonthermalChannel = Channel(
        paModel=PaModel(
            parameters={"modelName": "wiener", "width": 0}
        ),
        parameters={
            "thermalDutyCycle": 0.4,
            "activePowerThresholdDb": -60.0,
            "width": 0,
        },
    )
    assert np.isclose(
        nonthermalChannel.GetActualDutyCycle(periodicInput), 0.2
    )

    referenceThermalConfig = ThermalConfig(
        enabled=True,
        modelName="single_rc",
        sampleRateHz=100.0e3,
        thermalResistancesCPerW=(20.0,),
        thermalTimeConstantsSec=(5.0e-3,),
        thermalUpdateIntervalSamples=50,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        activePowerThresholdDb=-60.0,
        gainTemperatureCoefficientDbPerC=-0.03,
        maximumJunctionTemperatureC=200.0,
    )
    mismatchCases = (
        ("sampleRateHz", 101.0e3, 100.0e3),
        ("maximumOutputPowerDbm", 24.0, 25.0),
        ("activePowerThresholdDb", -50.0, -60.0),
    )
    expectedMetricNames = {
        "sampleRateHz": "sampleRateHz",
        "maximumOutputPowerDbm": "referenceOutputPowerDbm",
        "activePowerThresholdDb": "activePowerThresholdDb",
    }
    for channelParameterName, mismatchedValue, repairedValue in mismatchCases:
        mismatchPa = PaModel(
            parameters={
                "modelName": "wiener",
                "thermalConfig": referenceThermalConfig,
                "width": 0,
            }
        )
        mismatchChannel = Channel(
            paModel=mismatchPa,
            parameters={
                "sampleRateHz": 100.0e3,
                "maximumOutputPowerDbm": 25.0,
                "activePowerThresholdDb": -60.0,
                "thermalRunMode": "steady_state",
                "thermalDutyCycle": 0.4,
                "width": 0,
                channelParameterName: mismatchedValue,
            },
        )
        if channelParameterName == "sampleRateHz":
            try:
                mismatchChannel.CalibratePaInput(
                    periodicInput, outputPowerDbm=18.0
                )
            except ValueError as error:
                assert "sampleRateHz" in str(error)
            else:
                raise AssertionError(
                    "direct calibration accepted a thermal sample-rate "
                    "reference-plane mismatch"
                )
            try:
                mismatchChannel.GetLastCalibrationMetrics()
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "failed direct calibration committed diagnostics"
                )
        try:
            mismatchChannel.Process(periodicInput, outputPowerDbm=18.0)
        except ValueError as error:
            assert expectedMetricNames[channelParameterName] in str(error)
        else:
            raise AssertionError(
                "thermal reference-plane mismatch reached calibration: "
                f"{channelParameterName}"
            )
        try:
            mismatchChannel.GetLastCalibrationMetrics()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "reference-plane mismatch committed calibration diagnostics"
            )
        mismatchChannel.UpdateParameters(
            **{channelParameterName: repairedValue}
        )
        try:
            mismatchChannel.Process(periodicInput)
        except ValueError as error:
            assert "requires outputPowerDbm" in str(error)
        else:
            raise AssertionError(
                "failed reference-plane validation cached a power target"
            )

    # The second chain deliberately crosses its safety temperature after the
    # first chain has completed. With a zero mutual-heating matrix, the whole
    # MIMO period must still behave as one atomic thermal transaction.
    safeThermalConfig = replace(
        referenceThermalConfig,
        maximumJunctionTemperatureC=200.0,
    )
    failingThermalConfig = replace(
        referenceThermalConfig,
        maximumJunctionTemperatureC=25.000001,
    )
    failingMimoPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {
                    "modelName": "wiener",
                    "thermalConfig": safeThermalConfig,
                },
                {
                    "modelName": "wiener",
                    "thermalConfig": failingThermalConfig,
                },
            ),
            "thermalCouplingCPerW": (
                (0.0, 0.0),
                (0.0, 0.0),
            ),
            "width": 0,
        }
    )
    failureInput = np.column_stack(
        (
            np.full(1000, 0.4 + 0.0j, dtype=np.complex128),
            np.full(1000, 0.4 + 0.0j, dtype=np.complex128),
        )
    )
    metricsBeforeFailure = failingMimoPa.GetThermalMetrics()
    rmsBeforeFailure = failingMimoPa.GetOutputRmsPerChain()
    powerBeforeFailure = failingMimoPa.GetOutputPowerDbmPerChain()
    try:
        failingMimoPa.ProcessThermalPeriodFloating(
            failureInput,
            thermalRunMode="transient",
            thermalDutyCycle=0.5,
        )
    except RuntimeError as error:
        assert "maximumJunctionTemperatureC" in str(error)
    else:
        raise AssertionError(
            "the deliberately unsafe second thermal chain did not fail"
        )
    metricsAfterFailure = failingMimoPa.GetThermalMetrics()
    assert metricsAfterFailure == metricsBeforeFailure
    assert all(
        np.isclose(chainMetrics["elapsedTimeSec"], 0.0)
        for chainMetrics in metricsAfterFailure["chains"]
    )
    assert failingMimoPa.GetOutputRmsPerChain() == rmsBeforeFailure
    assert failingMimoPa.GetOutputPowerDbmPerChain() == powerBeforeFailure

    # Removing a live coupling matrix must also remove its retained external
    # temperature offsets. Otherwise a later nominally uncoupled period would
    # silently continue using stale heat from the preceding coupled solution.
    couplingResetMimoPa = MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {
                    "modelName": "wiener",
                    "thermalConfig": referenceThermalConfig,
                },
                {
                    "modelName": "wiener",
                    "thermalConfig": referenceThermalConfig,
                },
            ),
            "thermalCouplingCPerW": (
                (0.0, 1.0),
                (2.0, 0.0),
            ),
            "width": 0,
        }
    )
    couplingResetInput = np.column_stack(
        (periodicInput, 0.5 * periodicInput)
    )
    couplingResetMimoPa.ProcessThermalPeriodFloating(
        couplingResetInput,
        thermalRunMode="steady_state",
        thermalDutyCycle=0.4,
        steadyStateToleranceC=1.0e-5,
        maximumSteadyStateIterations=100,
    )
    coupledMetrics = couplingResetMimoPa.GetThermalMetrics()
    assert coupledMetrics["mutualHeating"]["steadyStateConverged"]
    assert all(
        chainMetrics["mutualHeatingTemperatureRiseC"] > 0.0
        for chainMetrics in coupledMetrics["chains"]
    )
    couplingResetMimoPa.UpdateParameters(
        thermalCouplingCPerW=(
            (0.0, 0.0),
            (0.0, 0.0),
        )
    )
    couplingResetMimoPa.ProcessThermalPeriodFloating(
        couplingResetInput,
        thermalRunMode="steady_state",
        thermalDutyCycle=0.4,
        steadyStateToleranceC=1.0e-5,
        maximumSteadyStateIterations=100,
    )
    uncoupledMetrics = couplingResetMimoPa.GetThermalMetrics()
    assert all(
        np.isclose(
            chainMetrics["mutualHeatingTemperatureRiseC"], 0.0
        )
        for chainMetrics in uncoupledMetrics["chains"]
    )
    assert uncoupledMetrics["mutualHeating"] == {
        "steadyStateConverged": True,
        "steadyStateIterations": 0,
        "steadyStateErrorC": 0.0,
    }
    assert all(
        chainMetrics["steadyStateConverged"]
        and chainMetrics["steadyStateErrorC"] <= 1.0e-5
        for chainMetrics in uncoupledMetrics["chains"]
    )

    # A transparent IQ wrapper must not hide the thermal protocol. The second
    # call omits its target and scales the raw input by five; equal outputs
    # therefore demonstrate that the cached target triggered fresh calibration
    # before a second live steady-state period was evaluated.
    iqWrappedPa = IQImbalancePA(
        PaModel(
            parameters={
                "modelName": "wiener",
                "thermalConfig": referenceThermalConfig,
                "width": 0,
            }
        ),
        directCoefficient=0.99 + 0.01j,
        imageCoefficient=0.025 - 0.01j,
    )
    iqChannel = Channel(
        paModel=iqWrappedPa,
        parameters={
            "sampleRateHz": 100.0e3,
            "maximumOutputPowerDbm": 25.0,
            "activePowerThresholdDb": -60.0,
            "thermalRunMode": "steady_state",
            "thermalDutyCycle": 0.4,
            "thermalSteadyStateToleranceC": 1.0e-6,
            "maximumThermalSteadyStateIterations": 100,
            "calibrationToleranceDb": 0.05,
            "width": 0,
        },
    )
    firstIqOutput, firstIqFeedbackOutput = iqChannel.Process(
        periodicInput, outputPowerDbm=18.0
    )
    firstIqMetrics = iqChannel.GetThermalMetrics()
    firstIqCalibration = iqChannel.GetLastCalibrationMetrics()
    secondIqOutput, secondIqFeedbackOutput = iqChannel.Process(
        0.2 * periodicInput
    )
    secondIqMetrics = iqChannel.GetThermalMetrics()
    secondIqCalibration = iqChannel.GetLastCalibrationMetrics()
    assert firstIqMetrics["thermalRunMode"] == "steady_state"
    assert secondIqMetrics["thermalRunMode"] == "steady_state"
    assert firstIqMetrics["scheduledIdleDurationSec"] > 0.0
    assert np.isclose(
        secondIqMetrics["elapsedTimeSec"],
        2.0 * firstIqMetrics["periodDurationSec"],
    )
    assert firstIqCalibration["targetOutputPowerDbmPerChain"] == (18.0,)
    assert secondIqCalibration["targetOutputPowerDbmPerChain"] == (18.0,)
    assert firstIqCalibration["converged"]
    assert secondIqCalibration["converged"]
    assert np.allclose(
        secondIqOutput,
        firstIqOutput,
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    assert np.allclose(
        secondIqFeedbackOutput,
        firstIqFeedbackOutput,
        rtol=1.0e-9,
        atol=1.0e-12,
    )


def CheckFeedbackIqPhasePairCalibration() -> None:
    """Verify 0/90-degree FB I/Q separation and Channel integration.

    Processing details:
        Algorithm: Solve ideal and measured nonideal phase-pair systems,
        prove that Tx/PA image content remains in the direct component, use
        symmetric ABBA acquisition to cancel linear complex drift, remove a
        known common receiver DC offset, fit and apply a widely-linear FIR,
        reject stale live-parameter coefficients, then exercise Channel
        phase-pair and cached-filter modes through floating SISO, fixed-point
        MIMO, thermal scheduling, and the dual-output ILC adapter. Finally,
        reject invalid modes, singular phase states, and filter controls.

    Returns:
        result: None. Assertions expose calibration, routing, state, shape,
            fixed-point, and validation regressions.
    """

    randomGenerator = np.random.default_rng(20260826)
    sampleCount = 2048
    sourceSignal = 0.16 * (
        randomGenerator.normal(size=sampleCount)
        + 1j * randomGenerator.normal(size=sampleCount)
    )

    # The direct component is the complete physical PA observation, including
    # a deliberate Tx/PA-created conjugate term and nonlinear envelope term.
    # Only the second component may contain the image created by the FB mixer.
    transmitterImageCoefficient = 0.11 - 0.025j
    physicalPaOutput = (
        sourceSignal
        + transmitterImageCoefficient * np.conj(sourceSignal)
        + (0.08 - 0.015j)
        * sourceSignal
        * np.abs(sourceSignal) ** 2
    )
    feedbackDirectCoefficient = 0.96 + 0.07j
    feedbackImageCoefficient = 0.075 - 0.028j
    expectedDirectSignal = feedbackDirectCoefficient * physicalPaOutput
    expectedImageSignal = (
        feedbackImageCoefficient * np.conj(physicalPaOutput)
    )
    idealZeroCapture = expectedDirectSignal + expectedImageSignal
    idealNinetyCapture = (
        1j * expectedDirectSignal - 1j * expectedImageSignal
    )
    idealCalibration = FeedbackIqCalibration(
        parameters={"width": 0}
    )
    separatedDirectSignal, separatedImageSignal = (
        idealCalibration.SeparatePhasePair(
            idealZeroCapture,
            idealNinetyCapture,
        )
    )
    assert np.allclose(
        separatedDirectSignal,
        expectedDirectSignal,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.allclose(
        separatedImageSignal,
        expectedImageSignal,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert not np.allclose(
        separatedDirectSignal,
        feedbackDirectCoefficient * sourceSignal,
    )

    # Measured switch responses need not have equal amplitude or an exact
    # ninety-degree separation. Matrix inversion must recover both components
    # and a known common receiver DC term must be removed before solving.
    zeroPhaseResponse = 0.93 * np.exp(1j * 0.08)
    ninetyPhaseResponse = 1.07 * np.exp(
        1j * (0.5 * np.pi - 0.06)
    )
    commonDcOffset = 0.021 - 0.014j
    nonidealZeroCapture = (
        zeroPhaseResponse * expectedDirectSignal
        + np.conj(zeroPhaseResponse) * expectedImageSignal
        + commonDcOffset
    )
    nonidealNinetyCapture = (
        ninetyPhaseResponse * expectedDirectSignal
        + np.conj(ninetyPhaseResponse) * expectedImageSignal
        + commonDcOffset
    )
    nonidealCalibration = FeedbackIqCalibration(
        parameters={
            "phaseResponses": (
                zeroPhaseResponse,
                ninetyPhaseResponse,
            ),
            "commonDcOffset": commonDcOffset,
            "width": 0,
        }
    )
    nonidealDirectSignal, nonidealImageSignal = (
        nonidealCalibration.SeparatePhasePair(
            nonidealZeroCapture,
            nonidealNinetyCapture,
        )
    )
    assert np.allclose(
        nonidealDirectSignal,
        expectedDirectSignal,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.allclose(
        nonidealImageSignal,
        expectedImageSignal,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    # A simple AB pair turns any capture-to-capture gain or phase change into
    # a false image. Symmetric A-B-B-A averaging gives both states the same
    # effective center time and cancels an exactly linear complex drift.
    driftSignal = sourceSignal[:512]
    complexDriftPerInterval = 0.012 + 0.017j
    zeroFirstCapture = (
        1.0 - 1.5 * complexDriftPerInterval
    ) * driftSignal
    ninetyFirstCapture = (
        1.0 - 0.5 * complexDriftPerInterval
    ) * 1j * driftSignal
    ninetySecondCapture = (
        1.0 + 0.5 * complexDriftPerInterval
    ) * 1j * driftSignal
    zeroSecondCapture = (
        1.0 + 1.5 * complexDriftPerInterval
    ) * driftSignal
    _, ordinaryAbImage = idealCalibration.SeparatePhasePair(
        zeroFirstCapture,
        ninetyFirstCapture,
    )
    abbaDirectSignal, abbaImageSignal = (
        idealCalibration.SeparateAbbaPhasePair(
            zeroFirstCapture,
            ninetyFirstCapture,
            ninetySecondCapture,
            zeroSecondCapture,
        )
    )
    ordinaryImageRms = float(
        np.sqrt(np.mean(np.abs(ordinaryAbImage) ** 2))
    )
    abbaImageRms = float(
        np.sqrt(np.mean(np.abs(abbaImageSignal) ** 2))
    )
    assert ordinaryImageRms > 1.0e-3
    assert abbaImageRms < ordinaryImageRms * 1.0e-10
    assert np.allclose(
        abbaDirectSignal,
        driftSignal,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    # A delayed image path requires a widely-linear FIR, not only one scalar
    # conjugate coefficient. The fitted filter must materially improve a new
    # single-state capture and expose stable diagnostic and tap snapshots.
    learningSignal = 0.14 * (
        randomGenerator.normal(size=4096)
        + 1j * randomGenerator.normal(size=4096)
    )
    feedbackImageTaps = np.asarray(
        (0.12 + 0.03j, -0.04 + 0.02j, 0.02 - 0.01j),
        dtype=np.complex128,
    )
    delayedImageSignal = np.convolve(
        np.conj(learningSignal),
        feedbackImageTaps,
        mode="full",
    )[: learningSignal.size]
    learningZeroCapture = learningSignal + delayedImageSignal
    learningNinetyCapture = (
        1j * learningSignal - 1j * delayedImageSignal
    )
    firCalibration = FeedbackIqCalibration(
        parameters={
            "filterLength": 9,
            "regularization": 1.0e-10,
            "width": 0,
        }
    )
    firMetrics = firCalibration.Calibrate(
        learningZeroCapture,
        learningNinetyCapture,
    )
    correctedLearningSignal = firCalibration.Apply(
        learningZeroCapture
    )
    rawResidualPower = float(
        np.mean(np.abs(learningZeroCapture - learningSignal) ** 2)
    )
    correctedResidualPower = float(
        np.mean(np.abs(correctedLearningSignal - learningSignal) ** 2)
    )
    directFilterTaps, conjugateFilterTaps = (
        firCalibration.GetFilterTaps()
    )
    assert firMetrics["calibrated"] is True
    assert firMetrics["sampleCount"] == learningSignal.size
    assert firMetrics["chainCount"] == 1
    assert firMetrics["filterLength"] == 9
    assert firMetrics["imageToDirectDb"] < -15.0
    assert firMetrics["fitNmseDb"] < -100.0
    assert directFilterTaps.shape == (9,)
    assert conjugateFilterTaps.shape == (9,)
    assert correctedResidualPower < rawResidualPower * 1.0e-8
    copiedMetrics = firCalibration.GetCalibrationMetrics()
    copiedDirectTaps, copiedConjugateTaps = (
        firCalibration.GetFilterTaps()
    )
    copiedDirectTaps[0] = 99.0 + 0.0j
    copiedConjugateTaps[0] = 99.0 + 0.0j
    assert copiedMetrics == firMetrics
    retainedDirectTaps, retainedConjugateTaps = (
        firCalibration.GetFilterTaps()
    )
    assert retainedDirectTaps[0] != copiedDirectTaps[0]
    assert retainedConjugateTaps[0] != copiedConjugateTaps[0]

    # Every live ChainMap input that changes the fitted numerical meaning must
    # make Apply reject stale taps. UpdateParameters follows the same rule.
    liveMutationCases = (
        (
            "phaseResponses",
            (1.0 + 0.0j, np.exp(1j * 1.42)),
        ),
        ("commonDcOffset", 0.001 + 0.002j),
        ("filterLength", 3),
        ("regularization", 2.0e-6),
        ("width", 16),
    )
    for parameterName, changedValue in liveMutationCases:
        liveParameters = {
            "phaseResponses": (1.0 + 0.0j, 0.0 + 1.0j),
            "commonDcOffset": 0.0 + 0.0j,
            "filterLength": 1,
            "regularization": 1.0e-6,
            "width": 0,
        }
        liveCalibration = FeedbackIqCalibration(
            parameters=liveParameters
        )
        liveCalibration.Calibrate(
            idealZeroCapture,
            idealNinetyCapture,
        )
        liveParameters[parameterName] = changedValue
        try:
            liveCalibration.Apply(idealZeroCapture)
        except RuntimeError as error:
            assert "Calibrate" in str(error) or "stale" in str(error)
        else:
            raise AssertionError(
                "live FeedbackIqCalibration change reused stale taps: "
                f"{parameterName}"
            )
        for staleAccessor in (
            liveCalibration.GetFilterTaps,
            liveCalibration.GetCalibrationMetrics,
        ):
            try:
                staleAccessor()
            except RuntimeError as error:
                assert "Calibrate" in str(error) or "stale" in str(error)
            else:
                raise AssertionError(
                    "live FeedbackIqCalibration change exposed stale "
                    f"artifacts: {parameterName}"
                )
    updatedCalibration = FeedbackIqCalibration(
        parameters={"width": 0}
    )
    updatedCalibration.Calibrate(
        idealZeroCapture,
        idealNinetyCapture,
    )
    updatedCalibration.UpdateParameters(regularization=3.0e-6)
    try:
        updatedCalibration.Apply(idealZeroCapture)
    except RuntimeError as error:
        assert "Calibrate" in str(error)
    else:
        raise AssertionError("UpdateParameters retained stale FB I/Q taps")

    # Channel must advance one physical PA thermal period and derive both raw
    # FB states from that one common PA output. Its phase-pair return is the
    # separated direct term, while diagnostics retain both raw observations.
    thermalSampleRateHz = 100.0e3
    thermalInput = 0.21 * (
        randomGenerator.normal(size=512)
        + 1j * randomGenerator.normal(size=512)
    )
    phasePairThermalConfig = ThermalConfig(
        enabled=True,
        modelName="single_rc",
        sampleRateHz=thermalSampleRateHz,
        thermalResistancesCPerW=(12.0,),
        thermalTimeConstantsSec=(0.02,),
        thermalUpdateIntervalSamples=32,
        idleDissipatedPowerW=0.0,
        referenceOutputPowerDbm=25.0,
        gainTemperatureCoefficientDbPerC=-0.01,
        maximumJunctionTemperatureC=150.0,
    )
    thermalPhasePairChannel = Channel(
        paModel=PaModel(
            parameters={
                "modelName": "wiener",
                "thermalConfig": phasePairThermalConfig,
                "width": 0,
            }
        ),
        parameters={
            "sampleMode": "fb",
            "sampleRateHz": thermalSampleRateHz,
            "thermalRunMode": "transient",
            "thermalDutyCycle": 1.0,
            "fbIqGainImbalanceDb": 1.4,
            "fbIqPhaseImbalanceDegrees": 6.0,
            "fbDcOffset": 0.012 - 0.008j,
            "fbPhasePairResponses": (
                zeroPhaseResponse,
                ninetyPhaseResponse,
            ),
            "fbThirdOrderCoefficient": 0.08 - 0.02j,
            "fbIqCompensationMode": "phase_pair",
            "fbIqCompensationFilterLength": 5,
            "fbIqCompensationRegularization": 1.0e-10,
            "width": 0,
        },
    )
    if thermalPhasePairChannel.paModel is None:
        raise AssertionError("thermal phase-pair Channel requires a PA")
    originalThermalProcessor = getattr(
        thermalPhasePairChannel.paModel,
        "ProcessThermalPeriodFloating",
    )
    originalFeedbackStateProcessor = (
        thermalPhasePairChannel.ApplyFeedbackChannelEffectsAtResponse
    )
    with patch.object(
        thermalPhasePairChannel.paModel,
        "ProcessThermalPeriodFloating",
        wraps=originalThermalProcessor,
    ) as thermalProcessorMock, patch.object(
        thermalPhasePairChannel,
        "ApplyFeedbackChannelEffectsAtResponse",
        wraps=originalFeedbackStateProcessor,
    ) as feedbackStateMock:
        thermalChannelOutput, thermalFeedbackOutput = (
            thermalPhasePairChannel.Process(thermalInput)
        )
        assert thermalProcessorMock.call_count == 1
        assert feedbackStateMock.call_count == 2
    thermalMetrics = thermalPhasePairChannel.GetThermalMetrics()
    rawZeroFeedback, rawNinetyFeedback = (
        thermalPhasePairChannel.GetLastFeedbackPhasePair()
    )
    channelCalibrationMetrics = (
        thermalPhasePairChannel.GetFeedbackIqCalibrationMetrics()
    )
    replayCalibration = FeedbackIqCalibration(
        parameters={
            "phaseResponses": (
                zeroPhaseResponse,
                ninetyPhaseResponse,
            ),
            "commonDcOffset": 0.012 - 0.008j,
            "filterLength": 5,
            "regularization": 1.0e-10,
            "width": 0,
        }
    )
    replayDirectFeedback, _ = replayCalibration.SeparatePhasePair(
        rawZeroFeedback,
        rawNinetyFeedback,
    )
    assert thermalChannelOutput.shape == thermalInput.shape
    assert thermalFeedbackOutput.shape == thermalInput.shape
    assert rawZeroFeedback.shape == thermalInput.shape
    assert rawNinetyFeedback.shape == thermalInput.shape
    assert channelCalibrationMetrics["calibrated"] is True
    assert np.allclose(
        thermalFeedbackOutput,
        replayDirectFeedback,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.isclose(
        thermalMetrics["elapsedTimeSec"],
        thermalInput.size / thermalSampleRateHz,
        rtol=0.0,
        atol=1.0e-15,
    )

    # Switching only the mode must reuse the fitted filter. A new filter-only
    # Channel has no calibration and must fail instead of silently using raw FB.
    savedRawZeroFeedback = rawZeroFeedback.copy()
    savedRawNinetyFeedback = rawNinetyFeedback.copy()
    savedChannelMetrics = dict(channelCalibrationMetrics)
    thermalPhasePairChannel.UpdateParameters(
        fbIqCompensationMode="filter"
    )
    _, cachedFilterOutput = thermalPhasePairChannel.Process(thermalInput)
    assert cachedFilterOutput.shape == thermalInput.shape
    assert np.all(np.isfinite(cachedFilterOutput))
    retainedRawPair = (
        thermalPhasePairChannel.GetLastFeedbackPhasePair()
    )
    assert np.array_equal(retainedRawPair[0], savedRawZeroFeedback)
    assert np.array_equal(retainedRawPair[1], savedRawNinetyFeedback)
    assert (
        thermalPhasePairChannel.GetFeedbackIqCalibrationMetrics()
        == savedChannelMetrics
    )
    uncalibratedFilterChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "sampleMode": "fb",
            "fbIqCompensationMode": "filter",
            "width": 0,
        },
    )
    try:
        uncalibratedFilterChannel.Process(thermalInput)
    except RuntimeError as error:
        assert "phase_pair" in str(error)
        assert "calibration" in str(error)
    else:
        raise AssertionError("uncalibrated filter mode was accepted")

    # DpdIlc must consume the corrected second member of the Channel tuple.
    # One iteration plus final replay is deterministic here, so the final
    # reported feedback must equal separation of Channel's retained raw pair.
    ilcReference = 0.18 * (
        randomGenerator.normal(size=1024)
        + 1j * randomGenerator.normal(size=1024)
    )
    ilcPhasePairChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "sampleMode": "fb",
            "sampleRateHz": 20.0e6,
            "fbIqGainImbalanceDb": 1.8,
            "fbIqPhaseImbalanceDegrees": -7.0,
            "fbIqCompensationMode": "phase_pair",
            "fbIqCompensationFilterLength": 5,
            "fbIqCompensationRegularization": 1.0e-10,
            "width": 0,
        },
    )
    phasePairIlcResult = RunScalarPIlc(
        ilcReference,
        ilcPhasePairChannel,
        ILCConfig(
            numIterations=1,
            learningRate=0.2,
            maxAmplitude=1.25,
        ),
        sampleRateHz=20.0e6,
    )
    ilcRawZero, ilcRawNinety = (
        ilcPhasePairChannel.GetLastFeedbackPhasePair()
    )
    ilcReplayCalibration = FeedbackIqCalibration(
        parameters={"width": 0}
    )
    expectedIlcFeedback, _ = (
        ilcReplayCalibration.SeparatePhasePair(
            ilcRawZero,
            ilcRawNinety,
        )
    )
    assert phasePairIlcResult.feedbackOutputSignal is not None
    assert np.allclose(
        phasePairIlcResult.feedbackOutputSignal,
        expectedIlcFeedback,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert not np.allclose(
        phasePairIlcResult.feedbackOutputSignal,
        ilcRawZero,
    )

    # Public width zero remains normalized floating data. Width 16 accepts and
    # returns integer-valued complex codes for a samples-by-two-chains matrix.
    floatingSisoChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "sampleMode": "fb",
            "fbIqCompensationMode": "phase_pair",
            "width": 0,
        },
    )
    floatingSisoOutput, floatingSisoFeedback = (
        floatingSisoChannel.Process(ilcReference[:256])
    )
    assert floatingSisoOutput.dtype == np.complex128
    assert floatingSisoFeedback.dtype == np.complex128
    assert np.max(np.abs(floatingSisoOutput)) < 1.0
    fixedSisoInput = FixedPoint(16).EncodeComplex(
        ilcReference[:256]
    )
    fixedSisoChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "sampleMode": "fb",
            "fbIqCompensationMode": "phase_pair",
            "width": 16,
        },
    )
    fixedSisoOutput, fixedSisoFeedback = fixedSisoChannel.Process(
        fixedSisoInput
    )
    for fixedSignal in (fixedSisoOutput, fixedSisoFeedback):
        assert fixedSignal.shape == fixedSisoInput.shape
        assert np.array_equal(fixedSignal.real, np.rint(fixedSignal.real))
        assert np.array_equal(fixedSignal.imag, np.rint(fixedSignal.imag))
    mimoFloatingInput = np.column_stack(
        (
            ilcReference[:256],
            0.71 * ilcReference[:256][::-1] * np.exp(0.31j),
        )
    )
    fixedFormat = FixedPoint(16)
    fixedMimoInput = fixedFormat.EncodeComplex(mimoFloatingInput)
    fixedMimoChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={
            "sampleMode": "fb",
            "fbIqGainImbalanceDb": 1.0,
            "fbIqPhaseImbalanceDegrees": 4.0,
            "fbIqCompensationMode": "phase_pair",
            "fbIqCompensationFilterLength": 3,
            "width": 16,
        },
    )
    fixedMimoOutput, fixedMimoFeedback = fixedMimoChannel.Process(
        fixedMimoInput
    )
    fixedRawZero, fixedRawNinety = (
        fixedMimoChannel.GetLastFeedbackPhasePair()
    )
    for fixedSignal in (
        fixedMimoOutput,
        fixedMimoFeedback,
        fixedRawZero,
        fixedRawNinety,
    ):
        assert fixedSignal.shape == fixedMimoInput.shape
        assert fixedSignal.dtype == np.complex128
        assert np.array_equal(fixedSignal.real, np.rint(fixedSignal.real))
        assert np.array_equal(fixedSignal.imag, np.rint(fixedSignal.imag))
        assert np.max(np.abs(fixedSignal.real)) <= 32767
        assert np.max(np.abs(fixedSignal.imag)) <= 32767
    assert fixedMimoChannel.GetFeedbackIqCalibrationMetrics()[
        "chainCount"
    ] == 2

    # Invalid mode names, incompatible forward acquisition, singular switch
    # responses, and illegal filter controls must fail with allowed values.
    invalidChannelCases = (
        (
            {"fbIqCompensationMode": "pair"},
            "fbIqCompensationMode",
            "'none', 'phase_pair', or 'filter'",
        ),
        (
            {
                "sampleMode": "forward",
                "fbIqCompensationMode": "phase_pair",
            },
            "fbIqCompensationMode",
            "sampleMode='fb'",
        ),
        (
            {
                "fbPhasePairResponses": (
                    1.0 + 0.0j,
                    2.0 + 0.0j,
                )
            },
            "fbPhasePairResponses",
            "0 or 180",
        ),
        (
            {"fbIqCompensationFilterLength": 0},
            "fbIqCompensationFilterLength",
            "[1, +inf)",
        ),
        (
            {"fbIqCompensationRegularization": 0.0},
            "fbIqCompensationRegularization",
            "(0, +inf)",
        ),
    )
    for invalidParameters, parameterName, allowedText in invalidChannelCases:
        try:
            Channel(parameters=invalidParameters)
        except (TypeError, ValueError) as error:
            assert parameterName in str(error)
            assert allowedText in str(error)
        else:
            raise AssertionError(
                f"invalid Channel setting accepted: {parameterName}"
            )
    invalidCalibrationCases = (
        ({"phaseResponses": (1.0 + 0.0j, -1.0 + 0.0j)}, "phaseResponses"),
        ({"commonDcOffset": complex(np.inf, 0.0)}, "commonDcOffset"),
        ({"filterLength": 0}, "filterLength"),
        ({"regularization": -1.0}, "regularization"),
    )
    for invalidParameters, parameterName in invalidCalibrationCases:
        try:
            FeedbackIqCalibration(parameters=invalidParameters)
        except (TypeError, ValueError) as error:
            assert parameterName in str(error)
        else:
            raise AssertionError(
                "invalid FeedbackIqCalibration setting accepted: "
                f"{parameterName}"
            )


def CheckFrequencySelectiveIqImbalance() -> None:
    """Verify causal frequency-selective Tx and feedback I/Q imbalance.

    Processing details:
        Algorithm: Compare the new widely-linear FIR stages with independent
        causal convolutions, recover their direct and mirror responses from two
        exact tones, prove legacy scalar fallback and Tx/FB isolation, exercise
        floating SISO plus fixed-point MIMO boundaries, validate live ChainMap
        updates and bad tap settings, then verify phase-pair separation, cached
        filter generalization, cache invalidation, and ILC feedback routing.

    Returns:
        result: None. Assertions expose response, configuration, compensation,
            representation, and integration regressions.
    """

    def ApplyReferenceIqFir(
        inputSignal: np.ndarray,
        directFirTaps: tuple[complex, ...],
        imageFirTaps: tuple[complex, ...],
        dcOffset: complex,
    ) -> np.ndarray:
        """Evaluate an independent causal widely-linear FIR reference.

        Processing details:
            Algorithm: Convert vectors to one-column matrices, convolve each
            physical chain independently with the supplied direct taps and its
            conjugate with the image taps, truncate both causal responses to
            the original record, add DC, and restore vector orientation.

        Args:
            inputSignal: SISO vector or samples-by-chains complex matrix.
            directFirTaps: Complete causal response of the desired branch.
            imageFirTaps: Complete causal response of the conjugate branch.
            dcOffset: Complex offset added after both filtered branches.

        Returns:
            result: Reference waveform with the same shape as the input.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        outputMatrix = np.empty_like(inputMatrix)
        directTaps = np.asarray(directFirTaps, dtype=np.complex128)
        imageTaps = np.asarray(imageFirTaps, dtype=np.complex128)
        for chainIndex in range(inputMatrix.shape[1]):
            inputColumn = inputMatrix[:, chainIndex]
            directOutput = np.convolve(
                inputColumn, directTaps, mode="full"
            )[: inputColumn.size]
            imageOutput = np.convolve(
                np.conj(inputColumn), imageTaps, mode="full"
            )[: inputColumn.size]
            outputMatrix[:, chainIndex] = (
                directOutput + imageOutput + dcOffset
            )
        return outputMatrix[:, 0] if inputWasVector else outputMatrix

    randomGenerator = np.random.default_rng(20260827)
    sourceSignal = 0.13 * (
        randomGenerator.normal(size=2048)
        + 1j * randomGenerator.normal(size=2048)
    )

    # None on both branches must preserve the historical scalar model. An
    # explicit one-tap pair containing the same alpha/beta values is equivalent,
    # while either branch can independently retain scalar fallback.
    legacyIqParameters = {
        "txIqGainImbalanceDb": 1.35,
        "txIqPhaseImbalanceDegrees": -5.5,
        "txDcOffset": 0.013 - 0.009j,
        "fbIqGainImbalanceDb": 1.35,
        "fbIqPhaseImbalanceDegrees": -5.5,
        "fbDcOffset": 0.013 - 0.009j,
        "width": 0,
    }
    legacyChannel = Channel(parameters=legacyIqParameters)
    legacyDirectCoefficient, legacyImageCoefficient = (
        legacyChannel.ResolveIqImbalanceCoefficients(1.35, -5.5)
    )
    expectedLegacyOutput = (
        legacyDirectCoefficient * sourceSignal
        + legacyImageCoefficient * np.conj(sourceSignal)
        + (0.013 - 0.009j)
    )
    assert legacyChannel.GetParameters()["txIqDirectFirTaps"] is None
    assert legacyChannel.GetParameters()["txIqImageFirTaps"] is None
    assert legacyChannel.GetParameters()["fbIqDirectFirTaps"] is None
    assert legacyChannel.GetParameters()["fbIqImageFirTaps"] is None
    assert np.allclose(
        legacyChannel.ApplyTransmitterIqImbalance(sourceSignal),
        expectedLegacyOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        legacyChannel.ApplyFeedbackIqImbalance(sourceSignal),
        expectedLegacyOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    explicitFlatChannel = Channel(
        parameters={
            **legacyIqParameters,
            "txIqDirectFirTaps": (legacyDirectCoefficient,),
            "txIqImageFirTaps": (legacyImageCoefficient,),
            "fbIqDirectFirTaps": (legacyDirectCoefficient,),
            "fbIqImageFirTaps": (legacyImageCoefficient,),
        }
    )
    assert np.allclose(
        explicitFlatChannel.ApplyTransmitterIqImbalance(sourceSignal),
        expectedLegacyOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        explicitFlatChannel.ApplyFeedbackIqImbalance(sourceSignal),
        expectedLegacyOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    hybridDirectTaps = (0.81 + 0.04j, 0.14 - 0.03j)
    hybridImageTaps = (0.07 - 0.02j, -0.025 + 0.01j)
    hybridChannel = Channel(
        parameters={
            **legacyIqParameters,
            "txIqDirectFirTaps": hybridDirectTaps,
            "fbIqImageFirTaps": hybridImageTaps,
        }
    )
    hybridTxDirect, hybridTxImage = (
        hybridChannel.TransmitterIqFilterTaps()
    )
    hybridFbDirect, hybridFbImage = hybridChannel.FeedbackIqFilterTaps()
    assert np.array_equal(hybridTxDirect, hybridDirectTaps)
    assert np.array_equal(hybridTxImage, (legacyImageCoefficient,))
    assert np.array_equal(hybridFbDirect, (legacyDirectCoefficient,))
    assert np.array_equal(hybridFbImage, hybridImageTaps)
    assert np.allclose(
        hybridChannel.ApplyTransmitterIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            hybridDirectTaps,
            (legacyImageCoefficient,),
            0.013 - 0.009j,
        ),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        hybridChannel.ApplyFeedbackIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            (legacyDirectCoefficient,),
            hybridImageTaps,
            0.013 - 0.009j,
        ),
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    # Explicit FIRs are complete branch responses, not corrections multiplied
    # by the legacy alpha/beta values. Use different Tx and FB responses and
    # compare both SISO and two-chain results with independent convolutions.
    txDirectTaps = (0.60 + 0.02j, 0.40 - 0.02j)
    txImageTaps = (0.08 - 0.01j, -0.06 + 0.02j)
    fbDirectTaps = (
        0.75 - 0.01j,
        -0.18 + 0.03j,
        0.0 + 0.08j,
    )
    fbImageTaps = (
        0.04 + 0.02j,
        0.07 - 0.01j,
        0.0 - 0.03j,
    )
    txDcOffset = 0.009 - 0.006j
    fbDcOffset = -0.011 + 0.007j
    selectiveChannel = Channel(
        parameters={
            "txIqGainImbalanceDb": 7.0,
            "txIqPhaseImbalanceDegrees": 23.0,
            "txIqDirectFirTaps": txDirectTaps,
            "txIqImageFirTaps": txImageTaps,
            "txDcOffset": txDcOffset,
            "fbIqGainImbalanceDb": -6.0,
            "fbIqPhaseImbalanceDegrees": -19.0,
            "fbIqDirectFirTaps": fbDirectTaps,
            "fbIqImageFirTaps": fbImageTaps,
            "fbDcOffset": fbDcOffset,
            "width": 0,
        }
    )
    expectedTxOutput = ApplyReferenceIqFir(
        sourceSignal,
        txDirectTaps,
        txImageTaps,
        txDcOffset,
    )
    expectedFbOutput = ApplyReferenceIqFir(
        sourceSignal,
        fbDirectTaps,
        fbImageTaps,
        fbDcOffset,
    )
    assert np.allclose(
        selectiveChannel.ApplyTransmitterIqImbalance(sourceSignal),
        expectedTxOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        selectiveChannel.ApplyFeedbackIqImbalance(sourceSignal),
        expectedFbOutput,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    sourceMatrix = np.column_stack(
        (
            sourceSignal,
            0.67 * sourceSignal[::-1] * np.exp(0.39j),
        )
    )
    expectedTxMatrix = ApplyReferenceIqFir(
        sourceMatrix,
        txDirectTaps,
        txImageTaps,
        txDcOffset,
    )
    expectedFbMatrix = ApplyReferenceIqFir(
        sourceMatrix,
        fbDirectTaps,
        fbImageTaps,
        fbDcOffset,
    )
    assert np.allclose(
        selectiveChannel.ApplyTransmitterIqImbalance(sourceMatrix),
        expectedTxMatrix,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        selectiveChannel.ApplyFeedbackIqImbalance(sourceMatrix),
        expectedFbMatrix,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    # Recover two wanted tones and both conjugate images by complex least
    # squares after the FIR startup transient. This checks the sign convention
    # Hdirect(+f) and Himage(-f), and proves that IRR is frequency selective.
    toneSampleCount = 4096
    toneIndices = np.arange(toneSampleCount, dtype=float)
    toneBinIndices = (128, 1792)
    toneRadians = tuple(
        2.0 * np.pi * toneBin / toneSampleCount
        for toneBin in toneBinIndices
    )
    toneAmplitudes = (0.17 + 0.04j, 0.11 - 0.06j)
    twoToneSignal = sum(
        toneAmplitude * np.exp(1j * toneRadian * toneIndices)
        for toneAmplitude, toneRadian in zip(
            toneAmplitudes, toneRadians
        )
    )
    toneStageCases = (
        (
            selectiveChannel.ApplyTransmitterIqImbalance,
            txDirectTaps,
            txImageTaps,
            txDcOffset,
        ),
        (
            selectiveChannel.ApplyFeedbackIqImbalance,
            fbDirectTaps,
            fbImageTaps,
            fbDcOffset,
        ),
    )
    for stageProcessor, directTaps, imageTaps, dcOffset in toneStageCases:
        toneOutput = stageProcessor(twoToneSignal)
        transientLength = max(len(directTaps), len(imageTaps)) - 1
        fitIndices = toneIndices[transientLength:]
        toneBasis = np.column_stack(
            (
                np.exp(1j * toneRadians[0] * fitIndices),
                np.exp(1j * toneRadians[1] * fitIndices),
                np.exp(-1j * toneRadians[0] * fitIndices),
                np.exp(-1j * toneRadians[1] * fitIndices),
                np.ones(fitIndices.size, dtype=np.complex128),
            )
        )
        measuredToneCoefficients = np.linalg.lstsq(
            toneBasis,
            toneOutput[transientLength:],
            rcond=None,
        )[0]
        directTapIndices = np.arange(len(directTaps), dtype=float)
        imageTapIndices = np.arange(len(imageTaps), dtype=float)
        expectedDirectCoefficients = tuple(
            toneAmplitude
            * np.sum(
                np.asarray(directTaps)
                * np.exp(-1j * toneRadian * directTapIndices)
            )
            for toneAmplitude, toneRadian in zip(
                toneAmplitudes, toneRadians
            )
        )
        expectedImageCoefficients = tuple(
            np.conj(toneAmplitude)
            * np.sum(
                np.asarray(imageTaps)
                * np.exp(1j * toneRadian * imageTapIndices)
            )
            for toneAmplitude, toneRadian in zip(
                toneAmplitudes, toneRadians
            )
        )
        expectedToneCoefficients = np.asarray(
            (
                *expectedDirectCoefficients,
                *expectedImageCoefficients,
                dcOffset,
            ),
            dtype=np.complex128,
        )
        assert np.allclose(
            measuredToneCoefficients,
            expectedToneCoefficients,
            rtol=1.0e-10,
            atol=1.0e-11,
        )
        toneIrrDb = tuple(
            20.0
            * np.log10(
                abs(expectedImageCoefficient)
                / abs(expectedDirectCoefficient)
            )
            for expectedDirectCoefficient, expectedImageCoefficient in zip(
                expectedDirectCoefficients,
                expectedImageCoefficients,
            )
        )
        assert max(toneIrrDb) < 0.0
        assert abs(toneIrrDb[1] - toneIrrDb[0]) > 5.0

    # Keep the four documented recommendation profiles, their plotted data,
    # and the Channel FIR convention tied to one numeric contract. Loading the
    # plotting module does not execute its __main__ block or regenerate PNGs.
    iqFigureDirectory = (
        GetProjectRoot() / "doc" / "images" / "channel_iq"
    )
    iqFigureNamespace = runpy.run_path(
        str(iqFigureDirectory / "GenerateIqIrrFigures.py")
    )
    buildRecommendedProfiles = iqFigureNamespace[
        "BuildRecommendedIqProfiles"
    ]
    calculateIqFrequencyResponses = iqFigureNamespace[
        "CalculateIqFrequencyResponses"
    ]
    calculateIrrDbCurve = iqFigureNamespace["CalculateIrrDbCurve"]
    recommendedProfiles = buildRecommendedProfiles()
    expectedProfileNames = (
        "flat_reference",
        "mild_frequency_selective",
        "moderate_edge_degradation",
        "severe_asymmetric_stress",
    )
    assert tuple(recommendedProfiles) == expectedProfileNames
    expectedProfileTaps = {
        "flat_reference": (
            (1.0 + 0.0j,),
            (0.010 + 0.0j,),
        ),
        "mild_frequency_selective": (
            (0.999 + 0.0j, 0.004 - 0.003j, -0.001 + 0.001j),
            (0.004 + 0.002j, -0.0015 + 0.001j, 0.0005 - 0.0005j),
        ),
        "moderate_edge_degradation": (
            (0.997 + 0.0j, 0.003 + 0.0j),
            (0.019 + 0.0j, -0.009 + 0.0j),
        ),
        "severe_asymmetric_stress": (
            (0.985 + 0.0j, 0.025 - 0.018j, -0.008 + 0.006j),
            (0.050 + 0.028j, -0.024 + 0.017j, 0.010 - 0.008j),
        ),
    }
    profileFrequencyMhz = np.asarray(
        (-40.0, -20.0, 0.0, 20.0, 40.0), dtype=float
    )
    profileFrequencyHz = profileFrequencyMhz * 1.0e6
    profileSampleRateHz = 80.0e6
    expectedProfileIrrDb = {
        "flat_reference": (
            -40.0,
            -40.0,
            -40.0,
            -40.0,
            -40.0,
        ),
        "mild_frequency_selective": (
            -44.35471790683431,
            -44.432977404199924,
            -48.18467329626209,
            -51.37063232715508,
            -44.35471790683431,
        ),
        "moderate_edge_degradation": (
            -31.004567061101884,
            -33.51971979464614,
            -40.0,
            -33.51971979464614,
            -31.004567061101884,
        ),
        "severe_asymmetric_stress": (
            -21.084376580467698,
            -21.740285051203543,
            -25.761005142336046,
            -31.504329797260844,
            -21.084376580467705,
        ),
    }
    calculatedProfileCsvValues = {}
    denseFrequencyHz = np.linspace(-40.0e6, 40.0e6, 2001)
    worstIrrDbByProfile = {}
    for profileName, profile in recommendedProfiles.items():
        directFirTaps = profile["directFirTaps"]
        imageFirTaps = profile["imageFirTaps"]
        assert (
            tuple(directFirTaps),
            tuple(imageFirTaps),
        ) == expectedProfileTaps[profileName]
        profileChannel = Channel(
            parameters={
                "txIqGainImbalanceDb": 0.0,
                "txIqPhaseImbalanceDegrees": 0.0,
                "txIqDirectFirTaps": directFirTaps,
                "txIqImageFirTaps": imageFirTaps,
                "width": 0,
            }
        )
        effectiveDirectTaps, effectiveImageTaps = (
            profileChannel.TransmitterIqFilterTaps()
        )
        assert np.array_equal(effectiveDirectTaps, directFirTaps)
        assert np.array_equal(effectiveImageTaps, imageFirTaps)
        keyPointIrrDb = np.asarray(
            calculateIrrDbCurve(
                profileFrequencyHz,
                profileSampleRateHz,
                effectiveDirectTaps,
                effectiveImageTaps,
            ),
            dtype=float,
        )
        assert np.allclose(
            keyPointIrrDb,
            expectedProfileIrrDb[profileName],
            rtol=0.0,
            atol=1.0e-10,
        )
        directResponse, imageAtMirrorResponse = (
            calculateIqFrequencyResponses(
                profileFrequencyHz,
                profileSampleRateHz,
                effectiveDirectTaps,
                effectiveImageTaps,
            )
        )
        calculatedProfileCsvValues[profileName] = np.column_stack(
            (
                keyPointIrrDb,
                20.0 * np.log10(np.abs(directResponse)),
                20.0 * np.log10(np.abs(imageAtMirrorResponse)),
            )
        )
        denseIrrDb = np.asarray(
            calculateIrrDbCurve(
                denseFrequencyHz,
                profileSampleRateHz,
                effectiveDirectTaps,
                effectiveImageTaps,
            ),
            dtype=float,
        )
        worstIrrDbByProfile[profileName] = float(np.max(denseIrrDb))

    assert (
        worstIrrDbByProfile["mild_frequency_selective"]
        < worstIrrDbByProfile["moderate_edge_degradation"]
        < worstIrrDbByProfile["severe_asymmetric_stress"]
    )

    iqCsvPath = iqFigureDirectory / "iq_irr_frequency_profiles.csv"
    iqCsvLines = iqCsvPath.read_text(encoding="utf-8").splitlines()
    expectedCsvHeaders = ["frequency_mhz"]
    for profileName in expectedProfileNames:
        expectedCsvHeaders.extend(
            (
                f"{profileName}_irr_db",
                f"{profileName}_direct_gain_db",
                f"{profileName}_image_gain_db",
            )
        )
    assert iqCsvLines[0].split(",") == expectedCsvHeaders
    iqCsvData = np.loadtxt(iqCsvPath, delimiter=",", skiprows=1)
    assert iqCsvData.shape == (2001, len(expectedCsvHeaders))
    assert np.allclose(
        iqCsvData[:, 0],
        np.linspace(-40.0, 40.0, 2001),
        rtol=0.0,
        atol=1.0e-12,
    )
    for frequencyIndex, frequencyMhz in enumerate(profileFrequencyMhz):
        csvRowIndex = int(round((frequencyMhz + 40.0) / 0.04))
        assert abs(iqCsvData[csvRowIndex, 0] - frequencyMhz) < 1.0e-12
        for profileIndex, profileName in enumerate(expectedProfileNames):
            irrColumnIndex = 1 + 3 * profileIndex
            assert np.allclose(
                iqCsvData[
                    csvRowIndex,
                    irrColumnIndex : irrColumnIndex + 3,
                ],
                calculatedProfileCsvValues[profileName][frequencyIndex],
                rtol=0.0,
                atol=5.0e-10,
            )

    # Returned tap arrays are defensive. Disabling either I/Q stage bypasses
    # its scalar settings, both FIRs, and DC as one atomic identity operation.
    returnedTxDirect, returnedTxImage = (
        selectiveChannel.TransmitterIqFilterTaps()
    )
    returnedFbDirect, returnedFbImage = (
        selectiveChannel.FeedbackIqFilterTaps()
    )
    returnedTxDirect[0] = 99.0 + 0.0j
    returnedTxImage[0] = 99.0 + 0.0j
    returnedFbDirect[0] = 99.0 + 0.0j
    returnedFbImage[0] = 99.0 + 0.0j
    retainedTxDirect, retainedTxImage = (
        selectiveChannel.TransmitterIqFilterTaps()
    )
    retainedFbDirect, retainedFbImage = (
        selectiveChannel.FeedbackIqFilterTaps()
    )
    assert np.array_equal(retainedTxDirect, txDirectTaps)
    assert np.array_equal(retainedTxImage, txImageTaps)
    assert np.array_equal(retainedFbDirect, fbDirectTaps)
    assert np.array_equal(retainedFbImage, fbImageTaps)
    disabledChannel = Channel(
        parameters={
            **selectiveChannel.GetParameters(),
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": False,
        }
    )
    disabledTxDirect, disabledTxImage = (
        disabledChannel.TransmitterIqFilterTaps()
    )
    disabledFbDirect, disabledFbImage = (
        disabledChannel.FeedbackIqFilterTaps()
    )
    assert np.array_equal(disabledTxDirect, (1.0 + 0.0j,))
    assert np.array_equal(disabledTxImage, (0.0 + 0.0j,))
    assert np.array_equal(disabledFbDirect, (1.0 + 0.0j,))
    assert np.array_equal(disabledFbImage, (0.0 + 0.0j,))
    assert np.array_equal(
        disabledChannel.ApplyTransmitterIqImbalance(sourceMatrix),
        sourceMatrix,
    )
    assert np.array_equal(
        disabledChannel.ApplyFeedbackIqImbalance(sourceMatrix),
        sourceMatrix,
    )

    # Tx FIRs precede the PA and therefore affect both observations. FB FIRs
    # affect only embedded feedback; forward mode must still return an exact
    # fbOut copy without evaluating the configured FB-specific fading.
    processInput = sourceSignal[:1024]
    commonProcessParameters = {
        "txIqDirectFirTaps": txDirectTaps,
        "txIqImageFirTaps": txImageTaps,
        "txDcOffset": txDcOffset,
        "fbIqDirectFirTaps": fbDirectTaps,
        "fbIqImageFirTaps": fbImageTaps,
        "fbDcOffset": fbDcOffset,
        "width": 0,
    }
    forwardSelectiveChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **commonProcessParameters,
            "sampleMode": "forward",
        },
    )
    feedbackSelectiveChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **commonProcessParameters,
            "sampleMode": "fb",
        },
    )
    forwardChannelOutput, forwardFeedbackOutput = (
        forwardSelectiveChannel.Process(processInput)
    )
    feedbackChannelOutput, embeddedFeedbackOutput = (
        feedbackSelectiveChannel.Process(processInput)
    )
    assert np.array_equal(forwardFeedbackOutput, forwardChannelOutput)
    assert np.allclose(
        feedbackChannelOutput,
        forwardChannelOutput,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.allclose(
        forwardSelectiveChannel.GetLastTransmitterOutput(),
        ApplyReferenceIqFir(
            processInput,
            txDirectTaps,
            txImageTaps,
            txDcOffset,
        ),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        embeddedFeedbackOutput,
        ApplyReferenceIqFir(
            feedbackChannelOutput,
            fbDirectTaps,
            fbImageTaps,
            fbDcOffset,
        ),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert not np.allclose(embeddedFeedbackOutput, feedbackChannelOutput)
    disabledProcessChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **commonProcessParameters,
            "sampleMode": "fb",
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": False,
        },
    )
    idealProcessChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={"sampleMode": "fb", "width": 0},
    )
    disabledChannelOutput, disabledFeedbackOutput = (
        disabledProcessChannel.Process(processInput)
    )
    idealChannelOutput, idealFeedbackOutput = idealProcessChannel.Process(
        processInput
    )
    assert np.allclose(disabledChannelOutput, idealChannelOutput)
    assert np.allclose(disabledFeedbackOutput, idealFeedbackOutput)

    # Floating matrices use one common response independently per chain. At a
    # public 16-bit boundary, FB processing must equal encode(reference(decode))
    # and Tx processing must expose the same normalized pre-PA matrix.
    fixedFormat = FixedPoint(16)
    fixedFloatingInput = sourceMatrix[:512]
    fixedInput = fixedFormat.EncodeComplex(fixedFloatingInput)
    fixedFeedbackChannel = Channel(
        parameters={
            "sampleMode": "fb",
            "fbIqDirectFirTaps": fbDirectTaps,
            "fbIqImageFirTaps": fbImageTaps,
            "fbDcOffset": fbDcOffset,
            "width": 16,
        }
    )
    fixedOutputFormat = FixedPoint(
        16, fixedFeedbackChannel.outputFullScaleAmplitude
    )
    fixedPaOutput = fixedOutputFormat.EncodeComplex(fixedFloatingInput)
    fixedFeedbackOutput = fixedFeedbackChannel.ProcessPaOutput(
        fixedPaOutput
    )
    decodedFixedInput = fixedOutputFormat.DecodeComplex(fixedPaOutput)
    expectedFixedFeedback = fixedOutputFormat.EncodeComplex(
        ApplyReferenceIqFir(
            decodedFixedInput,
            fbDirectTaps,
            fbImageTaps,
            fbDcOffset,
        )
    )
    assert np.array_equal(fixedFeedbackOutput, expectedFixedFeedback)

    widerPaOutputScale = 4.0
    mismatchedScalePa = PaModel(
        modelName="gmp",
        width=8,
        outputFullScaleAmplitude=widerPaOutputScale,
    )
    mismatchedScaleChannel = Channel(
        paModel=mismatchedScalePa,
        parameters={
            "sampleMode": "forward",
            "width": 16,
            "outputFullScaleAmplitude": 2.0,
        },
    )
    physicalPaOutput = np.array(
        [0.25 + 0.50j, -0.75 + 0.125j, 1.25 - 0.50j],
        dtype=np.complex128,
    )
    paOutputCodes = FixedPoint(
        8, widerPaOutputScale
    ).EncodeComplex(physicalPaOutput)
    expectedDecodedPaOutput = FixedPoint(
        8, widerPaOutputScale
    ).DecodeComplex(paOutputCodes)
    convertedChannelOutput = mismatchedScaleChannel.ProcessPaOutput(
        paOutputCodes
    )
    assert np.array_equal(
        convertedChannelOutput,
        FixedPoint(16, 2.0).EncodeComplex(expectedDecodedPaOutput),
    )

    fixedTxChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={
            "sampleMode": "forward",
            "txIqDirectFirTaps": txDirectTaps,
            "txIqImageFirTaps": txImageTaps,
            "txDcOffset": txDcOffset,
            "width": 16,
        },
    )
    fixedChannelOutput, fixedTxFeedbackOutput = fixedTxChannel.Process(
        fixedInput
    )
    decodedFixedTxInput = fixedFormat.DecodeComplex(fixedInput)
    expectedFixedTransmitter = ApplyReferenceIqFir(
        decodedFixedTxInput,
        txDirectTaps,
        txImageTaps,
        txDcOffset,
    )
    assert np.allclose(
        fixedTxChannel.GetLastTransmitterOutput(),
        expectedFixedTransmitter,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert fixedChannelOutput.shape == fixedInput.shape
    assert np.array_equal(fixedTxFeedbackOutput, fixedChannelOutput)
    assert np.array_equal(
        fixedChannelOutput.real, np.rint(fixedChannelOutput.real)
    )
    assert np.array_equal(
        fixedChannelOutput.imag, np.rint(fixedChannelOutput.imag)
    )

    # Caller-owned mappings remain live, while UpdateParameters has higher
    # priority and never mutates the lower layer. All four FIR names must take
    # effect immediately and invalid transactional updates must roll back.
    liveParameters = {
        "txIqDirectFirTaps": (1.0 + 0.0j,),
        "txIqImageFirTaps": (0.0 + 0.0j,),
        "fbIqDirectFirTaps": (1.0 + 0.0j,),
        "fbIqImageFirTaps": (0.0 + 0.0j,),
        "width": 0,
    }
    liveChannel = Channel(parameters=liveParameters)
    assert np.array_equal(
        liveChannel.ApplyTransmitterIqImbalance(sourceSignal), sourceSignal
    )
    liveTxDirectTaps = (0.92 + 0.01j, 0.06 - 0.02j)
    liveFbImageTaps = (0.05 - 0.01j, -0.015 + 0.02j)
    liveParameters["txIqDirectFirTaps"] = liveTxDirectTaps
    liveParameters["fbIqImageFirTaps"] = liveFbImageTaps
    assert np.allclose(
        liveChannel.ApplyTransmitterIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            liveTxDirectTaps,
            (0.0 + 0.0j,),
            0.0 + 0.0j,
        ),
    )
    assert np.allclose(
        liveChannel.ApplyFeedbackIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            (1.0 + 0.0j,),
            liveFbImageTaps,
            0.0 + 0.0j,
        ),
    )
    updatedTxImageTaps = (0.025 + 0.012j, 0.008 - 0.006j)
    updatedFbDirectTaps = (0.84 - 0.02j, 0.11 + 0.03j)
    liveChannel.UpdateParameters(
        txIqImageFirTaps=updatedTxImageTaps,
        fbIqDirectFirTaps=updatedFbDirectTaps,
    )
    assert liveParameters["txIqImageFirTaps"] == (0.0 + 0.0j,)
    assert liveParameters["fbIqDirectFirTaps"] == (1.0 + 0.0j,)
    assert np.allclose(
        liveChannel.ApplyTransmitterIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            liveTxDirectTaps,
            updatedTxImageTaps,
            0.0 + 0.0j,
        ),
    )
    assert np.allclose(
        liveChannel.ApplyFeedbackIqImbalance(sourceSignal),
        ApplyReferenceIqFir(
            sourceSignal,
            updatedFbDirectTaps,
            liveFbImageTaps,
            0.0 + 0.0j,
        ),
    )
    parametersBeforeInvalidUpdate = liveChannel.GetParameters()
    try:
        liveChannel.UpdateParameters(txIqImageFirTaps=())
    except ValueError as error:
        assert "txIqImageFirTaps" in str(error)
    else:
        raise AssertionError("empty live Tx image FIR was accepted")
    assert (
        liveChannel.GetParameters()["txIqImageFirTaps"]
        == parametersBeforeInvalidUpdate["txIqImageFirTaps"]
    )

    invalidFirCases = (
        ("txIqDirectFirTaps", "not taps", TypeError),
        ("txIqImageFirTaps", (), ValueError),
        ("fbIqDirectFirTaps", ((1.0,), (0.5,)), ValueError),
        ("fbIqImageFirTaps", (complex(np.inf, 0.0),), ValueError),
    )
    for parameterName, invalidValue, expectedErrorType in invalidFirCases:
        try:
            Channel(parameters={parameterName: invalidValue})
        except (TypeError, ValueError) as error:
            assert isinstance(error, expectedErrorType)
            assert parameterName in str(error)
            assert "nonempty one-dimensional sequence" in str(error)
        else:
            raise AssertionError(
                f"invalid frequency-selective FIR accepted: {parameterName}"
            )

    # Phase-pair separation must preserve the direct FB FIR while removing the
    # frequency-selective image. Its learned single-state inverse must
    # generalize to an independent record, and either live or explicit FB tap
    # changes must invalidate the cached filter before another transmission.
    calibrationImageTaps = (
        0.11 + 0.03j,
        -0.045 + 0.025j,
        0.018 - 0.012j,
    )
    calibrationDcOffset = 0.012 - 0.008j
    calibrationParameters = {
        "sampleMode": "fb",
        "fbIqDirectFirTaps": (1.0 + 0.0j,),
        "fbIqImageFirTaps": calibrationImageTaps,
        "fbDcOffset": calibrationDcOffset,
        "fbIqCompensationMode": "phase_pair",
        "fbIqCompensationFilterLength": 17,
        "fbIqCompensationRegularization": 1.0e-10,
        "width": 0,
    }
    calibrationChannel = Channel(parameters=calibrationParameters)
    calibrationSignal = 0.14 * (
        randomGenerator.normal(size=8192)
        + 1j * randomGenerator.normal(size=8192)
    )
    separatedFeedback = calibrationChannel.ProcessPaOutput(
        calibrationSignal
    )
    rawPhaseZero, rawPhaseNinety = (
        calibrationChannel.GetLastFeedbackPhasePair()
    )
    replayCalibration = FeedbackIqCalibration(
        parameters={
            "commonDcOffset": calibrationDcOffset,
            "filterLength": 17,
            "regularization": 1.0e-10,
            "width": 0,
        }
    )
    replayDirect, replayImage = replayCalibration.SeparatePhasePair(
        rawPhaseZero,
        rawPhaseNinety,
    )
    expectedCalibrationImage = ApplyReferenceIqFir(
        calibrationSignal,
        (0.0 + 0.0j,),
        calibrationImageTaps,
        0.0 + 0.0j,
    )
    assert np.allclose(
        separatedFeedback,
        calibrationSignal,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.allclose(replayDirect, calibrationSignal)
    assert np.allclose(replayImage, expectedCalibrationImage)
    assert calibrationChannel.GetFeedbackIqCalibrationMetrics()[
        "fitNmseDb"
    ] < -100.0
    calibrationChannel.UpdateParameters(
        fbIqCompensationMode="filter"
    )
    independentSignal = 0.14 * (
        randomGenerator.normal(size=4096)
        + 1j * randomGenerator.normal(size=4096)
    )
    correctedIndependentSignal = calibrationChannel.ProcessPaOutput(
        independentSignal
    )
    rawCalibrationChannel = Channel(
        parameters={
            **calibrationParameters,
            "fbIqCompensationMode": "none",
        }
    )
    rawIndependentSignal = rawCalibrationChannel.ProcessPaOutput(
        independentSignal
    )
    rawIndependentError = float(
        np.mean(np.abs(rawIndependentSignal - independentSignal) ** 2)
    )
    correctedIndependentError = float(
        np.mean(
            np.abs(correctedIndependentSignal - independentSignal) ** 2
        )
    )
    assert correctedIndependentError < rawIndependentError * 1.0e-8

    calibrationParameters["fbIqDirectFirTaps"] = (
        0.98 + 0.0j,
        0.01 + 0.0j,
    )
    try:
        calibrationChannel.ProcessPaOutput(independentSignal)
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("live FB direct FIR reused stale calibration")
    calibrationParameters["fbIqDirectFirTaps"] = (1.0 + 0.0j,)
    calibrationChannel.UpdateParameters(
        fbIqCompensationMode="phase_pair"
    )
    calibrationChannel.ProcessPaOutput(calibrationSignal)
    calibrationChannel.UpdateParameters(
        fbIqCompensationMode="filter"
    )
    calibrationChannel.UpdateParameters(
        fbIqImageFirTaps=(0.09 + 0.02j, -0.03 + 0.01j)
    )
    assert calibrationParameters["fbIqImageFirTaps"] == (
        calibrationImageTaps
    )
    try:
        calibrationChannel.ProcessPaOutput(independentSignal)
    except RuntimeError as error:
        assert "calibration" in str(error)
    else:
        raise AssertionError("updated FB image FIR retained calibration")

    # Frequency-domain ILC must receive the phase-pair or cached-filter fbOut,
    # not the raw frequency-selective mirror. Direct FB response is identity,
    # so corrected feedback should match the same-round forward observation.
    ilcParameters = {
        **calibrationParameters,
        "fbIqDirectFirTaps": (1.0 + 0.0j,),
        "fbIqImageFirTaps": calibrationImageTaps,
        "fbIqCompensationMode": "phase_pair",
    }
    ilcChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters=ilcParameters,
    )
    ilcReference = 0.12 * (
        randomGenerator.normal(size=1024)
        + 1j * randomGenerator.normal(size=1024)
    )
    ilcConfig = ILCConfig(
        numIterations=2,
        learningRate=0.2,
        maxAmplitude=1.25,
    )
    phasePairIlcResult = RunFrequencyDomainIlc(
        ilcReference,
        ilcChannel,
        sampleRateHz=20.0e6,
        channelBandwidthHz=8.0e6,
        config=ilcConfig,
    )
    assert len(phasePairIlcResult.history) == 2
    for iterationResult in phasePairIlcResult.history:
        assert iterationResult.feedbackOutputSignal is not None
        feedbackMismatch = float(
            np.mean(
                np.abs(
                    iterationResult.feedbackOutputSignal
                    - iterationResult.outputSignal
                )
                ** 2
            )
        )
        outputPower = max(
            float(np.mean(np.abs(iterationResult.outputSignal) ** 2)),
            np.finfo(float).tiny,
        )
        assert feedbackMismatch / outputPower < 1.0e-24
    ilcChannel.UpdateParameters(fbIqCompensationMode="filter")
    filterIlcReference = 0.12 * (
        randomGenerator.normal(size=1024)
        + 1j * randomGenerator.normal(size=1024)
    )
    filterIlcResult = RunFrequencyDomainIlc(
        filterIlcReference,
        ilcChannel,
        sampleRateHz=20.0e6,
        channelBandwidthHz=8.0e6,
        config=ilcConfig,
    )
    assert len(filterIlcResult.history) == 2
    for iterationResult in filterIlcResult.history:
        assert iterationResult.feedbackOutputSignal is not None
        feedbackMismatch = float(
            np.mean(
                np.abs(
                    iterationResult.feedbackOutputSignal
                    - iterationResult.outputSignal
                )
                ** 2
            )
        )
        outputPower = max(
            float(np.mean(np.abs(iterationResult.outputSignal) ** 2)),
            np.finfo(float).tiny,
        )
        assert feedbackMismatch / outputPower < 1.0e-16
    rawIlcChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **ilcParameters,
            "fbIqCompensationMode": "none",
        },
    )
    rawIlcChannelOutput, rawIlcFeedbackOutput = rawIlcChannel.Process(
        ilcReference
    )
    rawIlcMismatch = float(
        np.mean(np.abs(rawIlcFeedbackOutput - rawIlcChannelOutput) ** 2)
    )
    rawIlcOutputPower = float(
        np.mean(np.abs(rawIlcChannelOutput) ** 2)
    )
    assert rawIlcMismatch / rawIlcOutputPower > 1.0e-3

    # Scalar small-signal diagnostics use the direct FIR response at DC; image
    # branches remain excluded because one scalar cannot represent a mirror.
    gainPaModel = PaModel(modelName="wiener", width=0)
    gainChannel = Channel(
        paModel=gainPaModel,
        parameters={
            "sampleMode": "fb",
            "txIqDirectFirTaps": txDirectTaps,
            "txIqImageFirTaps": txImageTaps,
            "fbIqDirectFirTaps": fbDirectTaps,
            "fbIqImageFirTaps": fbImageTaps,
            "width": 0,
        },
    )
    expectedSmallSignalGain = (
        np.sum(txDirectTaps)
        * gainPaModel.SmallSignalGain()
        * np.sum(fbDirectTaps)
    )
    assert np.allclose(
        gainChannel.SmallSignalGain(), expectedSmallSignalGain
    )


def CheckChannelIqEnableControls() -> None:
    """Verify independent Tx and feedback I/Q-stage enable controls.

    Processing details:
        Algorithm: Preserve the historical enabled-by-default behavior, prove
        that each False switch bypasses its gain mismatch, phase error, and DC
        offset as one atomic stage, exercise independent Tx-only and
        feedback-only paths in both sample modes, repeat the bypass through a
        fixed-point two-chain PA, and reject every non-boolean switch value
        with a diagnostic that states the allowed values.

    Returns:
        result: None. Assertions enforce I/Q-stage gating at every interface.
    """

    testSignal = np.asarray(
        (0.21 + 0.13j, -0.37 + 0.08j, 0.04 - 0.29j, -0.11 - 0.18j),
        dtype=np.complex128,
    )
    configuredIqParameters = {
        "txIqGainImbalanceDb": 1.75,
        "txIqPhaseImbalanceDegrees": 8.0,
        "txDcOffset": 0.025 - 0.017j,
        "fbIqGainImbalanceDb": -1.25,
        "fbIqPhaseImbalanceDegrees": -6.0,
        "fbDcOffset": -0.019 + 0.011j,
        "width": 0,
    }

    # True defaults preserve the pre-switch API: omitting either enable name
    # must produce exactly the same coefficients and samples as explicit True.
    implicitEnabledChannel = Channel(parameters=configuredIqParameters)
    explicitEnabledChannel = Channel(
        parameters={
            **configuredIqParameters,
            "txIqImbalanceEnabled": True,
            "fbIqImbalanceEnabled": True,
        }
    )
    implicitParameters = implicitEnabledChannel.GetParameters()
    assert implicitParameters["txIqImbalanceEnabled"] is True
    assert implicitParameters["fbIqImbalanceEnabled"] is True
    assert (
        implicitEnabledChannel.TransmitterIqCoefficients()
        == explicitEnabledChannel.TransmitterIqCoefficients()
    )
    assert (
        implicitEnabledChannel.FeedbackIqCoefficients()
        == explicitEnabledChannel.FeedbackIqCoefficients()
    )
    assert np.array_equal(
        implicitEnabledChannel.ApplyTransmitterIqImbalance(testSignal),
        explicitEnabledChannel.ApplyTransmitterIqImbalance(testSignal),
    )
    assert np.array_equal(
        implicitEnabledChannel.ApplyFeedbackIqImbalance(testSignal),
        explicitEnabledChannel.ApplyFeedbackIqImbalance(testSignal),
    )

    # False atomically bypasses all three terms. Deliberately large finite
    # values ensure a partial bypass could not accidentally pass this check.
    disabledIqChannel = Channel(
        parameters={
            **configuredIqParameters,
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": False,
        }
    )
    assert disabledIqChannel.TransmitterIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert disabledIqChannel.FeedbackIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert np.array_equal(
        disabledIqChannel.ApplyTransmitterIqImbalance(testSignal),
        testSignal,
    )
    assert np.array_equal(
        disabledIqChannel.ApplyFeedbackIqImbalance(testSignal),
        testSignal,
    )
    assert not np.array_equal(
        explicitEnabledChannel.ApplyTransmitterIqImbalance(testSignal),
        testSignal,
    )
    assert not np.array_equal(
        explicitEnabledChannel.ApplyFeedbackIqImbalance(testSignal),
        testSignal,
    )

    # A feedback-only defect is invisible to forward instrument sampling but
    # visible to embedded feedback sampling. Disabling Tx must not disable FB.
    feedbackOnlyForwardChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **configuredIqParameters,
            "sampleMode": "forward",
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": True,
        }
    )
    feedbackOnlyEmbeddedChannel = Channel(
        parameters={
            **configuredIqParameters,
            "sampleMode": "fb",
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": True,
        }
    )
    assert np.array_equal(
        feedbackOnlyForwardChannel.ProcessPaOutput(testSignal),
        testSignal,
    )
    expectedFeedbackOutput = (
        feedbackOnlyEmbeddedChannel.ApplyFeedbackIqImbalance(testSignal)
    )
    assert np.allclose(
        feedbackOnlyEmbeddedChannel.ProcessPaOutput(testSignal),
        expectedFeedbackOutput,
    )
    assert not np.allclose(expectedFeedbackOutput, testSignal)
    (
        feedbackOnlyForwardOutput,
        feedbackOnlyForwardReturnedFeedback,
    ) = feedbackOnlyForwardChannel.Process(testSignal)
    assert np.array_equal(
        feedbackOnlyForwardReturnedFeedback,
        feedbackOnlyForwardOutput,
    )

    # Conversely, a Tx-only defect drives the PA in both sample modes. An
    # independently disabled FB stage makes the two observations identical.
    txOnlyForwardChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **configuredIqParameters,
            "sampleMode": "forward",
            "txIqImbalanceEnabled": True,
            "fbIqImbalanceEnabled": False,
        },
    )
    txOnlyEmbeddedChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **configuredIqParameters,
            "sampleMode": "fb",
            "txIqImbalanceEnabled": True,
            "fbIqImbalanceEnabled": False,
        },
    )
    (
        txOnlyForwardOutput,
        txOnlyForwardFeedbackOutput,
    ) = txOnlyForwardChannel.Process(testSignal)
    (
        txOnlyEmbeddedOutput,
        txOnlyEmbeddedFeedbackOutput,
    ) = txOnlyEmbeddedChannel.Process(testSignal)
    assert np.allclose(txOnlyForwardFeedbackOutput, txOnlyForwardOutput)
    assert np.allclose(txOnlyEmbeddedFeedbackOutput, txOnlyEmbeddedOutput)
    assert np.allclose(txOnlyEmbeddedOutput, txOnlyForwardOutput)
    assert np.allclose(
        txOnlyForwardChannel.GetLastTransmitterOutput(),
        txOnlyForwardChannel.ApplyTransmitterIqImbalance(testSignal),
    )
    assert not np.allclose(
        txOnlyForwardChannel.GetLastTransmitterOutput(), testSignal
    )

    # High-priority UpdateParameters overrides must gate the live ChainMap
    # immediately without mutating the caller-owned lower-priority mapping.
    # Exercise all three useful states through the same Channel and PA objects
    # so stale coefficients or cached processing decisions cannot pass.
    liveIqParameters = {
        **configuredIqParameters,
        "sampleMode": "fb",
        "txIqImbalanceEnabled": False,
        "fbIqImbalanceEnabled": False,
    }
    liveIqChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters=liveIqParameters,
    )
    allDisabledOutput, allDisabledFeedbackOutput = liveIqChannel.Process(
        testSignal
    )
    assert np.array_equal(allDisabledFeedbackOutput, allDisabledOutput)
    assert liveIqChannel.TransmitterIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert liveIqChannel.FeedbackIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert np.array_equal(
        liveIqChannel.GetLastTransmitterOutput(), testSignal
    )

    liveIqChannel.UpdateParameters(txIqImbalanceEnabled=True)
    assert liveIqParameters["txIqImbalanceEnabled"] is False
    assert liveIqChannel.GetParameters()["txIqImbalanceEnabled"] is True
    assert liveIqChannel.TransmitterIqCoefficients() != (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert liveIqChannel.FeedbackIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    txOnlyLiveOutput, txOnlyLiveFeedbackOutput = liveIqChannel.Process(
        testSignal
    )
    txOnlyLiveDrive = liveIqChannel.ApplyTransmitterIqImbalance(testSignal)
    expectedTxOnlyLiveOutput = liveIqChannel.paModel.Process(
        txOnlyLiveDrive
    )
    assert np.allclose(txOnlyLiveOutput, expectedTxOnlyLiveOutput)
    assert np.allclose(txOnlyLiveFeedbackOutput, expectedTxOnlyLiveOutput)
    assert np.allclose(
        liveIqChannel.GetLastTransmitterOutput(), txOnlyLiveDrive
    )
    assert not np.allclose(txOnlyLiveOutput, allDisabledOutput)

    liveIqChannel.UpdateParameters(
        txIqImbalanceEnabled=False,
        fbIqImbalanceEnabled=True,
    )
    liveResolvedParameters = liveIqChannel.GetParameters()
    assert liveResolvedParameters["txIqImbalanceEnabled"] is False
    assert liveResolvedParameters["fbIqImbalanceEnabled"] is True
    assert liveIqParameters["fbIqImbalanceEnabled"] is False
    assert liveIqChannel.TransmitterIqCoefficients() == (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    assert liveIqChannel.FeedbackIqCoefficients() != (
        1.0 + 0.0j,
        0.0 + 0.0j,
    )
    (
        feedbackOnlyLiveChannelOutput,
        feedbackOnlyLiveOutput,
    ) = liveIqChannel.Process(testSignal)
    expectedFeedbackOnlyLiveOutput = (
        liveIqChannel.ApplyFeedbackIqImbalance(allDisabledOutput)
    )
    assert np.allclose(
        feedbackOnlyLiveOutput, expectedFeedbackOnlyLiveOutput
    )
    assert np.allclose(feedbackOnlyLiveChannelOutput, allDisabledOutput)
    assert np.array_equal(
        liveIqChannel.GetLastTransmitterOutput(), testSignal
    )
    assert not np.allclose(feedbackOnlyLiveOutput, allDisabledOutput)

    # The disabled stages must remain transparent for samples-by-chains data
    # after public 16-bit decoding and re-encoding around a MIMO PA bank.
    mimoFloatingInput = np.column_stack(
        (testSignal, 0.63 * testSignal[::-1] * np.exp(0.37j))
    )
    fixedFormat = FixedPoint(16)
    mimoFixedInput = fixedFormat.EncodeComplex(mimoFloatingInput)
    idealFixedChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={"sampleMode": "fb", "width": 16},
    )
    disabledFixedChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={
            **configuredIqParameters,
            "sampleMode": "fb",
            "txIqImbalanceEnabled": False,
            "fbIqImbalanceEnabled": False,
            "width": 16,
        },
    )
    enabledFixedChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={
            **configuredIqParameters,
            "sampleMode": "fb",
            "txIqImbalanceEnabled": True,
            "fbIqImbalanceEnabled": True,
            "width": 16,
        },
    )
    idealFixedOutput, idealFixedFeedbackOutput = idealFixedChannel.Process(
        mimoFixedInput
    )
    (
        disabledFixedOutput,
        disabledFixedFeedbackOutput,
    ) = disabledFixedChannel.Process(mimoFixedInput)
    enabledFixedOutput, enabledFixedFeedbackOutput = (
        enabledFixedChannel.Process(mimoFixedInput)
    )
    assert disabledFixedOutput.shape == mimoFixedInput.shape
    assert disabledFixedOutput.dtype == np.complex128
    assert np.array_equal(disabledFixedOutput, idealFixedOutput)
    assert np.array_equal(
        disabledFixedFeedbackOutput, idealFixedFeedbackOutput
    )
    assert np.array_equal(disabledFixedFeedbackOutput, disabledFixedOutput)
    assert not np.array_equal(enabledFixedOutput, idealFixedOutput)
    assert not np.array_equal(
        enabledFixedFeedbackOutput, idealFixedFeedbackOutput
    )
    assert np.array_equal(
        disabledFixedChannel.GetLastTransmitterOutput(),
        fixedFormat.DecodeComplex(mimoFixedInput),
    )

    # Boolean switches intentionally reject integer, string, None, and NumPy
    # scalar lookalikes so configuration mistakes cannot silently enable IQ.
    for parameterName in (
        "txIqImbalanceEnabled",
        "fbIqImbalanceEnabled",
    ):
        for invalidValue in (0, 1, "False", None, np.bool_(False)):
            try:
                Channel(parameters={parameterName: invalidValue})
            except TypeError as error:
                errorText = str(error)
                assert parameterName in errorText
                assert "invalid type" in errorText
                assert "Allowed values: True or False" in errorText
            else:
                raise AssertionError(
                    f"{parameterName} accepted non-boolean "
                    f"{invalidValue!r}"
                )


def CheckChannelDualOutputContract() -> None:
    """Verify parallel channel/feedback outputs and their consumers.

    Processing details:
        Algorithm: Compare ideal and feedback-impaired calibrated Channels,
        prove that feedback-only settings leave the clean PA power target and
        forward observation unchanged, check that one dual-output call advances
        exactly one thermal waveform period, enforce integer-code boundaries on
        both fixed-point outputs, then run a synthetic dual-output ILC plant.
        The synthetic plant keeps its forward output perfect while distorting
        feedback, so any ILC MSE or waveform update must come from ``fbOut``;
        independent history analysis must still obtain EVM from ``chOut``.

    Returns:
        result: None. Assertions enforce reference-plane and routing contracts.
    """

    calibrationSignal = np.r_[
        np.zeros(48, dtype=np.complex128),
        0.31
        * np.exp(
            1j
            * 2.0
            * np.pi
            * np.arange(1024, dtype=float)
            / 31.0
        ),
        np.zeros(64, dtype=np.complex128),
    ]
    commonChannelParameters = {
        "sampleMode": "fb",
        "maximumOutputPowerDbm": 25.0,
        "calibrationToleranceDb": 0.05,
        "width": 0,
    }
    idealChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters=commonChannelParameters,
    )
    impairedFeedbackChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            **commonChannelParameters,
            "fbGainDb": -4.0,
            "fbPhaseDegrees": 31.0,
            "fbFirTaps": (1.0 + 0.0j, 0.18 - 0.07j),
            "fbIntegerDelaySamples": 2,
            "fbThirdOrderCoefficient": -0.08 + 0.01j,
        },
    )
    idealOutput, idealFeedbackOutput = idealChannel.Process(
        calibrationSignal,
        outputPowerDbm=17.0,
    )
    impairedOutput, impairedFeedbackOutput = (
        impairedFeedbackChannel.Process(
            calibrationSignal,
            outputPowerDbm=17.0,
        )
    )
    idealCalibrationMetrics = idealChannel.GetLastCalibrationMetrics()
    impairedCalibrationMetrics = (
        impairedFeedbackChannel.GetLastCalibrationMetrics()
    )
    assert np.allclose(idealOutput, idealFeedbackOutput)
    assert np.allclose(impairedOutput, idealOutput)
    assert not np.allclose(impairedFeedbackOutput, impairedOutput)
    assert np.allclose(
        impairedFeedbackOutput,
        impairedFeedbackChannel.ApplyFeedbackChannelEffects(
            impairedFeedbackChannel.GetLastPaOutput()
        ),
    )
    assert abs(
        idealCalibrationMetrics["measuredOutputPowerDbmPerChain"][0]
        - 17.0
    ) <= 0.05
    assert abs(
        impairedCalibrationMetrics[
            "measuredOutputPowerDbmPerChain"
        ][0]
        - 17.0
    ) <= 0.05
    assert np.allclose(
        impairedCalibrationMetrics["analogDriveDbPerChain"],
        idealCalibrationMetrics["analogDriveDbPerChain"],
    )

    # Forward sampling intentionally does not expose the embedded receiver.
    # Even with every feedback-only impairment enabled and a nonzero noise
    # request, fbOut must be an exact sample-for-sample copy of chOut. Using
    # array_equal also catches accidental independent noise generation.
    forwardModeChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": 20.0e6,
            "maximumOutputPowerDbm": 25.0,
            "fbGainDb": 7.0,
            "fbPhaseDegrees": -43.0,
            "fbFirTaps": (1.0 + 0.0j, -0.21 + 0.13j),
            "fbIntegerDelaySamples": 3,
            "fbFractionalDelaySamples": 0.2,
            "fbCarrierFrequencyOffsetHz": 2500.0,
            "fbSamplingFrequencyOffsetPpm": -35.0,
            "fbIqGainImbalanceDb": 2.5,
            "fbIqPhaseImbalanceDegrees": 9.0,
            "fbDcOffset": 0.04 - 0.03j,
            "fbThirdOrderCoefficient": -0.12 + 0.03j,
            "fbClipAmplitude": 0.45,
            "fbAdcWidth": 8,
            "fbAdcFullScale": 0.55,
            "noiseSnrDb": 34.0,
            "randomSeed": 817,
            "width": 0,
        },
    )
    forwardModeOutput, forwardModeFeedbackOutput = (
        forwardModeChannel.Process(calibrationSignal)
    )
    assert np.array_equal(forwardModeFeedbackOutput, forwardModeOutput)

    # Splitting the common post-PA waveform must not evaluate or heat the PA
    # twice. The elapsed time therefore advances by exactly N / Fs seconds.
    thermalSampleRateHz = 100.0e3
    thermalInput = np.full(200, 0.23 + 0.04j, dtype=np.complex128)
    thermalPa = PaModel(
        parameters={
            "modelName": "wiener",
            "thermalConfig": ThermalConfig.Recommended(
                "single_rc",
                sampleRateHz=thermalSampleRateHz,
                thermalUpdateIntervalSamples=20,
            ),
            "width": 0,
        }
    )
    thermalChannel = Channel(
        paModel=thermalPa,
        parameters={
            "sampleMode": "fb",
            "sampleRateHz": thermalSampleRateHz,
            "thermalRunMode": "transient",
            "fbGainDb": -3.0,
            "width": 0,
        },
    )
    elapsedTimeBeforeSec = float(
        thermalPa.GetThermalMetrics()["elapsedTimeSec"]
    )
    thermalOutput, thermalFeedbackOutput = thermalChannel.Process(
        thermalInput
    )
    elapsedTimeAfterSec = float(
        thermalPa.GetThermalMetrics()["elapsedTimeSec"]
    )
    assert thermalOutput.shape == thermalInput.shape
    assert thermalFeedbackOutput.shape == thermalInput.shape
    assert not np.allclose(thermalFeedbackOutput, thermalOutput)
    assert np.isclose(
        elapsedTimeAfterSec - elapsedTimeBeforeSec,
        thermalInput.size / thermalSampleRateHz,
    )

    # Both public fixed-point branches retain the same complex-array type and
    # integer I/Q-code convention even though the feedback values differ.
    fixedFormat = FixedPoint(16)
    fixedInput = fixedFormat.EncodeComplex(
        np.asarray(
            (0.20 + 0.10j, -0.30 + 0.04j, 0.08 - 0.21j),
            dtype=np.complex128,
        )
    )
    fixedChannel = Channel(
        paModel=PaModel(modelName="wiener", width=16),
        parameters={
            "sampleMode": "fb",
            "fbGainDb": -2.0,
            "fbPhaseDegrees": -17.0,
            "width": 16,
        },
    )
    fixedOutput, fixedFeedbackOutput = fixedChannel.Process(fixedInput)
    for publicOutput in (fixedOutput, fixedFeedbackOutput):
        assert publicOutput.shape == fixedInput.shape
        assert publicOutput.dtype == np.complex128
        assert np.array_equal(publicOutput.real, np.rint(publicOutput.real))
        assert np.array_equal(publicOutput.imag, np.rint(publicOutput.imag))
    assert not np.array_equal(fixedFeedbackOutput, fixedOutput)

    # Repeat the forward-copy contract at the public fixed-point MIMO
    # boundary. The severe feedback settings and receiver noise are ignored in
    # this mode, while both returned matrices retain integer-valued I/Q codes.
    fixedMimoInput = np.column_stack(
        (
            fixedInput,
            fixedFormat.EncodeComplex(
                np.asarray(
                    (-0.14 + 0.23j, 0.09 - 0.27j, 0.31 + 0.02j),
                    dtype=np.complex128,
                )
            ),
        )
    )
    fixedMimoForwardChannel = Channel(
        paModel=MimoPaModel(
            parameters={"numTransmitChains": 2, "width": 0}
        ),
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": 20.0e6,
            "maximumOutputPowerDbm": 25.0,
            "fbGainDb": -9.0,
            "fbPhaseDegrees": 71.0,
            "fbFirTaps": (1.0 + 0.0j, 0.18j),
            "fbIntegerDelaySamples": 1,
            "fbIqGainImbalanceDb": -3.0,
            "fbIqPhaseImbalanceDegrees": 12.0,
            "fbThirdOrderCoefficient": 0.15 - 0.04j,
            "fbAdcWidth": 7,
            "fbAdcFullScale": 0.7,
            "noiseAmpMv": 10.0,
            "randomSeed": 823,
            "width": 16,
        },
    )
    fixedMimoOutput, fixedMimoFeedbackOutput = (
        fixedMimoForwardChannel.Process(fixedMimoInput)
    )
    assert fixedMimoOutput.shape == fixedMimoInput.shape
    assert fixedMimoOutput.dtype == np.complex128
    assert np.array_equal(fixedMimoFeedbackOutput, fixedMimoOutput)
    assert np.array_equal(
        fixedMimoOutput.real, np.rint(fixedMimoOutput.real)
    )
    assert np.array_equal(
        fixedMimoOutput.imag, np.rint(fixedMimoOutput.imag)
    )

    class SyntheticDualOutputPlant:
        """Expose a perfect forward observation and nonlinear feedback."""

        def __init__(self, channelReference: np.ndarray) -> None:
            """Retain the perfect forward waveform used by RF analysis.

            Processing details:
                Algorithm: Copy the immutable target and expose a floating
                public boundary compatible with the normalized ILC adapter.

            Args:
                channelReference: Ideal waveform returned on the forward path.

            Returns:
                result: None. The deterministic dual-output plant is ready.
            """

            self.channelReference = np.asarray(
                channelReference, dtype=np.complex128
            ).copy()
            self.sampleMode = "fb"
            self.width = 0

        def ProcessOutputPathsFloating(
            self, inputSignal: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return perfect ``chOut`` and input-dependent distorted fbOut.

            Processing details:
                Algorithm: Keep the forward waveform independent of the ILC
                drive while applying a memoryless cubic response to feedback.
                A learner incorrectly using chOut would see zero error and
                could not change its second-round input.

            Args:
                inputSignal: Current normalized ILC drive waveform.

            Returns:
                result: Forward reference and nonlinear feedback arrays.
            """

            complexInput = np.asarray(inputSignal, dtype=np.complex128)
            feedbackOutput = (
                0.72 * complexInput
                + 0.48 * complexInput * np.abs(complexInput) ** 2
            )
            return self.channelReference.copy(), feedbackOutput

    wifiWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=5,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        seed=619,
        width=0,
    ).Generate()
    ilcReference = 0.20 * wifiWaveform.samples
    dualOutputConfig = ILCConfig(
        numIterations=2,
        learningRate=0.25,
        maxAmplitude=1.25,
    )
    ilcMethodRunners = (
        (
            "scalar",
            lambda selectedPlant: RunScalarPIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                wifiWaveform.sampleRateHz,
            ),
        ),
        (
            "complex",
            lambda selectedPlant: RunComplexGainIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                wifiWaveform.sampleRateHz,
            ),
        ),
        (
            "fir",
            lambda selectedPlant: RunFirIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                firLength=9,
                sampleRateHz=wifiWaveform.sampleRateHz,
            ),
        ),
        (
            "frequency",
            lambda selectedPlant: RunFrequencyDomainIlc(
                ilcReference,
                selectedPlant,
                wifiWaveform.sampleRateHz,
                wifiWaveform.bandwidthHz,
                dualOutputConfig,
            ),
        ),
        (
            "directional",
            lambda selectedPlant: RunDirectionalGaussNewtonIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                finiteDifferenceRms=1.0e-3,
                sampleRateHz=wifiWaveform.sampleRateHz,
            ),
        ),
        (
            "parameter",
            lambda selectedPlant: RunParameterDomainIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                nonlinearOrders=(1, 3),
                memoryDepth=2,
                sampleRateHz=wifiWaveform.sampleRateHz,
            ),
        ),
        (
            "augmented",
            lambda selectedPlant: RunAugmentedIqIlc(
                ilcReference,
                selectedPlant,
                dualOutputConfig,
                wifiWaveform.sampleRateHz,
            ),
        ),
    )
    methodResults = {}
    for methodName, methodRunner in ilcMethodRunners:
        methodResult = methodRunner(
            SyntheticDualOutputPlant(ilcReference)
        )
        methodResults[methodName] = methodResult
        assert len(methodResult.history) == 2
        assert np.array_equal(methodResult.outputSignal, ilcReference)
        assert methodResult.feedbackOutputSignal is not None
        assert not np.allclose(
            methodResult.feedbackOutputSignal,
            methodResult.outputSignal,
        )
        firstIteration = methodResult.history[0]
        secondIteration = methodResult.history[1]
        assert np.array_equal(firstIteration.outputSignal, ilcReference)
        assert firstIteration.feedbackOutputSignal is not None
        assert not np.allclose(
            firstIteration.feedbackOutputSignal,
            firstIteration.outputSignal,
        )
        explicitlyAlignedFeedback = SigProc(
            ilcReference,
            wifiWaveform.sampleRateHz,
        ).Process(firstIteration.feedbackOutputSignal).processedSignal
        expectedFeedbackMse = float(
            np.mean(
                np.abs(ilcReference - explicitlyAlignedFeedback) ** 2
            )
        )
        assert np.isclose(firstIteration.mse, expectedFeedbackMse), methodName
        assert firstIteration.mse > 0.0, methodName
        assert not np.allclose(
            secondIteration.inputSignal,
            firstIteration.inputSignal,
        ), methodName

    ilcResult = methodResults["frequency"]
    firstIteration = ilcResult.history[0]

    ilcAnalysis = Analysis(
        ilcReference,
        wifiWaveform,
        parameters={"width": 0},
    )
    analyzedHistory = ilcAnalysis.AnalyzeIlcHistory(ilcResult.history)
    directChannelMetrics = ilcAnalysis.Analyze(firstIteration.outputSignal)
    directFeedbackMetrics = ilcAnalysis.Analyze(
        firstIteration.feedbackOutputSignal
    )
    assert np.isclose(
        analyzedHistory.history[0].evmDb,
        directChannelMetrics["evmDb"],
    )
    assert (
        analyzedHistory.history[0].evmDb
        < directFeedbackMetrics["evmDb"]
    )
    assert np.array_equal(analyzedHistory.bestOutputSignal, ilcReference)


def CheckChannelModel() -> None:
    """Verify both sample modes, feedback impairments, noise, and fixed point.

    Processing details:
        Algorithm: Prove transmitter I/Q error precedes the PA while forward
        sampling bypasses embedded feedback I/Q error, exercise deterministic
        feedback gain/FIR/delay/CFO and combined nonlinear/IQ/ADC effects,
        compare amplitude and power noise controls, verify active-burst SNR for
        SISO/MIMO, require repeatable random state, validate hidden PA power
        calibration, and reject invalid settings.

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

    # Tx I/Q imbalance is a physical forward-path impairment before the PA,
    # while feedback I/Q imbalance is an observation-only receiver defect.
    txPaModel = PaModel(modelName="wiener", width=0)
    txIqChannel = Channel(
        paModel=txPaModel,
        parameters={
            "sampleMode": "forward",
            "txIqGainImbalanceDb": 1.2,
            "txIqPhaseImbalanceDegrees": 5.0,
            "txDcOffset": 0.015 - 0.01j,
            "fbIqGainImbalanceDb": 6.0,
            "fbIqPhaseImbalanceDegrees": 20.0,
            "fbDcOffset": 0.2 + 0.1j,
            "width": 0,
        },
    )
    txDirectCoefficient, txImageCoefficient = (
        txIqChannel.TransmitterIqCoefficients()
    )
    expectedTxOutput = (
        txDirectCoefficient * testSignal
        + txImageCoefficient * np.conj(testSignal)
        + (0.015 - 0.01j)
    )
    assert np.allclose(
        txIqChannel.ApplyTransmitterIqImbalance(testSignal),
        expectedTxOutput,
    )
    expectedForwardOutput = txPaModel.Process(expectedTxOutput)
    txChannelOutput, txFeedbackOutput = txIqChannel.Process(testSignal)
    assert np.allclose(
        txChannelOutput,
        expectedForwardOutput,
    )
    assert np.array_equal(txFeedbackOutput, txChannelOutput)
    assert np.allclose(
        txIqChannel.GetLastTransmitterOutput(), expectedTxOutput
    )
    assert np.allclose(
        txIqChannel.GetLastActualPaInput(), expectedTxOutput
    )
    assert np.allclose(
        txIqChannel.SmallSignalGain(),
        txDirectCoefficient * txPaModel.SmallSignalGain(),
    )
    assert np.allclose(
        txIqChannel.ProcessPaOutput(txPaModel.Process(testSignal)),
        txPaModel.Process(testSignal),
    )

    combinedIqChannel = Channel(
        paModel=txPaModel,
        parameters={
            "sampleMode": "fb",
            "txIqGainImbalanceDb": 1.2,
            "txIqPhaseImbalanceDegrees": 5.0,
            "txDcOffset": 0.015 - 0.01j,
            "fbIqGainImbalanceDb": -0.8,
            "fbIqPhaseImbalanceDegrees": -3.0,
            "fbDcOffset": -0.02 + 0.005j,
            "width": 0,
        },
    )
    combinedTxOutput = combinedIqChannel.ApplyTransmitterIqImbalance(
        testSignal
    )
    combinedPaOutput = txPaModel.Process(combinedTxOutput)
    expectedCombinedIqOutput = combinedIqChannel.ApplyFeedbackIqImbalance(
        combinedPaOutput
    )
    combinedChannelOutput, combinedFeedbackOutput = (
        combinedIqChannel.Process(testSignal)
    )
    assert np.allclose(combinedChannelOutput, combinedPaOutput)
    assert np.allclose(
        combinedFeedbackOutput,
        expectedCombinedIqOutput,
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
    channelOutput, feedbackOutput = paChannel.Process(testSignal)
    assert np.allclose(channelOutput, expectedOutput)
    assert np.allclose(feedbackOutput, expectedOutput)
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
            "txIqGainImbalanceDb": 0.7,
            "txIqPhaseImbalanceDegrees": 2.5,
            "width": 0,
        },
    )
    calibratedOutput, calibratedFeedbackOutput = calibratedChannel.Process(
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
    assert np.array_equal(calibratedFeedbackOutput, calibratedOutput)
    assert calibratedChannel.GetLastPaInput().shape == rawBurst.shape
    assert not np.allclose(
        calibratedChannel.GetLastPaInput(),
        calibratedChannel.GetLastActualPaInput(),
    )

    # Third-party thermal PA adapters must expose an atomic suspend/restore
    # pair. Reject an incomplete adapter before its suspend method can mutate
    # state, and restore a complete adapter even when calibration processing
    # raises an exception.
    class IncompleteThermalPa:
        """Expose one invalid half of the thermal transaction interface."""

        def __init__(self) -> None:
            """Initialize an observable unsuspended state.

            Processing details:
                Algorithm: Store one Boolean marker so the test can prove that
                Channel validates the complete transaction interface before
                invoking the available suspend method.

            Returns:
                result: None. The synthetic PA starts unsuspended.
            """

            self.suspended = False

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Return the supplied signal unchanged.

            Processing details:
                Algorithm: Convert the trial waveform to the public complex
                type without modifying its samples, providing the minimum PA
                processing contract required by Channel.

            Args:
                inputSignal: Calibration trial waveform.

            Returns:
                result: Complex copy or view of the supplied waveform.
            """

            return np.asarray(inputSignal, dtype=np.complex128)

        def SuspendThermalModel(self) -> bool:
            """Mutate state if Channel incorrectly accepts this adapter.

            Processing details:
                Algorithm: Mark the adapter suspended and return a trivial
                snapshot. Correct Channel validation rejects the adapter before
                this deliberately incomplete transaction can run.

            Returns:
                result: True as a synthetic prior thermal-state snapshot.
            """

            self.suspended = True
            return True

    incompleteThermalPa = IncompleteThermalPa()
    incompleteThermalChannel = Channel(
        paModel=incompleteThermalPa,
        parameters={"width": 0},
    )
    try:
        incompleteThermalChannel.Process(
            rawBurst,
            outputPowerDbm=18.0,
        )
    except TypeError as error:
        assert "SuspendThermalModel and RestoreThermalModel" in str(error)
    else:
        raise AssertionError("an incomplete PA thermal interface was accepted")
    assert not incompleteThermalPa.suspended

    class FailingThermalPa:
        """Raise during calibration to verify transactional restoration."""

        def __init__(self) -> None:
            """Initialize one active thermal-state marker.

            Processing details:
                Algorithm: Record an active Boolean state and a restoration
                counter so the test can audit the Channel finally block after
                an intentional PA observation failure.

            Returns:
                result: None. The synthetic thermal state starts active.
            """

            self.thermalActive = True
            self.restoreCount = 0

        def Process(self, inputSignal: np.ndarray) -> np.ndarray:
            """Represent a failed instrument or PA observation.

            Processing details:
                Algorithm: Raise a deterministic RuntimeError whenever the
                calibration loop attempts to evaluate the synthetic PA.

            Args:
                inputSignal: Unused calibration trial waveform.

            Returns:
                result: No waveform is returned because processing fails.
            """

            raise RuntimeError("intentional calibration observation failure")

        def SuspendThermalModel(self) -> bool:
            """Save and suspend the synthetic thermal-state marker.

            Processing details:
                Algorithm: Copy the active marker, clear it during calibration,
                and return the copy as the complete synthetic snapshot.

            Returns:
                result: Boolean thermal state that existed before calibration.
            """

            thermalSnapshot = self.thermalActive
            self.thermalActive = False
            return thermalSnapshot

        def RestoreThermalModel(self, thermalSnapshot: bool) -> None:
            """Restore the marker and count the completed transaction.

            Processing details:
                Algorithm: Replace the active marker with the supplied snapshot
                and increment a counter exactly once per restoration call.

            Args:
                thermalSnapshot: Boolean state returned by suspension.

            Returns:
                result: None. The synthetic thermal state is restored.
            """

            self.thermalActive = thermalSnapshot
            self.restoreCount += 1

    failingThermalPa = FailingThermalPa()
    failingThermalChannel = Channel(
        paModel=failingThermalPa,
        parameters={"width": 0},
    )
    try:
        failingThermalChannel.Process(
            rawBurst,
            outputPowerDbm=18.0,
        )
    except RuntimeError as error:
        assert "intentional calibration observation failure" in str(error)
    else:
        raise AssertionError("a failing PA calibration unexpectedly completed")
    assert failingThermalPa.thermalActive
    assert failingThermalPa.restoreCount == 1

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
    mimoOutput, mimoFeedbackOutput = mimoChannel.Process(
        mimoRawBurst,
        outputPowerDbm=(17.0, 19.0),
    )
    mimoCalibrationMetrics = (
        mimoChannel.GetLastCalibrationMetrics()
    )
    assert mimoOutput.shape == mimoRawBurst.shape
    assert mimoFeedbackOutput.shape == mimoRawBurst.shape
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

    # The rated 25 dBm PA range must include a 20 dBm operating point even
    # when an EHT waveform and the default GMP are exposed through a 16-bit
    # interface. The hidden drive belongs after DAC decoding, while all public
    # arrays must remain integer-valued I/Q codes.
    fixedWifiWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=10,
        seed=101,
        width=16,
    ).Generate()
    fixedGmpChannel = Channel(
        paModel=PaModel(modelName="gmp", width=16),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.25,
            "maximumCalibrationIterations": 60,
            "width": 16,
        },
    )
    fixedGmpOutput, fixedGmpFeedbackOutput = fixedGmpChannel.Process(
        fixedWifiWaveform.samples,
        outputPowerDbm=20.0,
    )
    fixedGmpMetrics = fixedGmpChannel.GetLastCalibrationMetrics()
    assert fixedGmpMetrics["converged"] is True
    assert np.array_equal(
        fixedGmpOutput,
        fixedGmpChannel.GetLastPaOutput(),
    )
    assert fixedGmpMetrics["targetOutputPowerDbmPerChain"] == (20.0,)
    assert abs(
        fixedGmpMetrics["measuredOutputPowerDbmPerChain"][0] - 20.0
    ) <= 0.25
    assert 1 <= fixedGmpMetrics["iterationCount"] <= 60
    fixedWifiDecoded = FixedPoint(16).DecodeComplex(
        fixedWifiWaveform.samples
    )
    fixedWifiHeadroomRestoreDb = -20.0 * np.log10(
        np.sqrt(np.mean(np.abs(fixedWifiDecoded) ** 2))
    )
    residualAnalogDriveDb = (
        fixedGmpMetrics["analogDriveDbPerChain"][0]
        - fixedWifiHeadroomRestoreDb
    )
    assert fixedWifiHeadroomRestoreDb > 0.0
    assert abs(residualAnalogDriveDb) < 3.0
    actualFixedGmpInput = fixedGmpChannel.GetLastActualPaInput()
    assert np.max(np.abs(actualFixedGmpInput)) <= 2.0
    for publicWaveform in (
        fixedGmpChannel.GetLastPaInput(),
        fixedGmpChannel.GetLastPaOutput(),
        fixedGmpOutput,
        fixedGmpFeedbackOutput,
    ):
        assert publicWaveform.dtype == np.complex128
        assert np.array_equal(
            publicWaveform.real, np.rint(publicWaveform.real)
        )
        assert np.array_equal(
            publicWaveform.imag, np.rint(publicWaveform.imag)
        )
    fixedGmpOutputMetrics = Analysis(
        fixedGmpOutput,
        transmittedSignal=fixedGmpOutput,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "activePowerThresholdDb": -60.0,
            "activeGapToleranceSamples": 16,
            "width": 16,
            "outputFullScaleAmplitude": (
                fixedGmpChannel.outputFullScaleAmplitude
            ),
        },
    ).Analyze()
    assert abs(fixedGmpOutputMetrics["outputPowerDbm"] - 20.0) <= 0.25
    decodedFixedGmpInput = FixedPoint(16).DecodeComplex(
        fixedGmpChannel.GetLastPaInput()
    )
    assert np.max(np.abs(decodedFixedGmpInput.real)) < 0.51
    assert np.max(np.abs(decodedFixedGmpInput.imag)) < 0.51

    defaultHeadroomPeak = float(
        np.max(
            np.maximum(
                np.abs(decodedFixedGmpInput.real),
                np.abs(decodedFixedGmpInput.imag),
            )
        )
    )
    fixedGmpChannel.UpdateParameters(
        calibrationDigitalHeadroomDb=9.0
    )
    (
        largerHeadroomOutput,
        largerHeadroomFeedbackOutput,
    ) = fixedGmpChannel.Process(
        fixedWifiWaveform.samples,
        outputPowerDbm=20.0,
    )
    largerHeadroomMetrics = fixedGmpChannel.GetLastCalibrationMetrics()
    largerHeadroomInput = fixedGmpChannel.GetLastPaInput()
    decodedLargerHeadroomInput = FixedPoint(16).DecodeComplex(
        largerHeadroomInput
    )
    largerHeadroomPeak = float(
        np.max(
            np.maximum(
                np.abs(decodedLargerHeadroomInput.real),
                np.abs(decodedLargerHeadroomInput.imag),
            )
        )
    )
    assert largerHeadroomPeak < 0.75 * defaultHeadroomPeak
    assert abs(
        largerHeadroomMetrics["measuredOutputPowerDbmPerChain"][0]
        - 20.0
    ) <= 0.25
    assert largerHeadroomFeedbackOutput.shape == largerHeadroomOutput.shape
    largerHeadroomOutputMetrics = Analysis(
        largerHeadroomOutput,
        transmittedSignal=largerHeadroomOutput,
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "width": 16,
            "outputFullScaleAmplitude": (
                fixedGmpChannel.outputFullScaleAmplitude
            ),
        },
    ).Analyze()
    assert abs(
        largerHeadroomOutputMetrics["outputPowerDbm"] - 20.0
    ) <= 0.25

    # A failed replacement request must not commit its trial analog drive.
    # Reusing the previously accepted public codes must therefore reproduce
    # the same deterministic channel output after the failure.
    (
        stableOutputBeforeFailure,
        stableFeedbackBeforeFailure,
    ) = fixedGmpChannel.Process(largerHeadroomInput)
    fixedGmpChannel.UpdateParameters(
        calibrationToleranceDb=1.0e-6,
        maximumCalibrationIterations=1,
    )
    try:
        fixedGmpChannel.Process(
            fixedWifiWaveform.samples,
            outputPowerDbm=24.0,
        )
    except RuntimeError:
        failedReplacementMetrics = (
            fixedGmpChannel.GetLastCalibrationMetrics()
        )
        assert failedReplacementMetrics["converged"] is False
    else:
        raise AssertionError("a one-trial replacement unexpectedly converged")
    (
        stableOutputAfterFailure,
        stableFeedbackAfterFailure,
    ) = fixedGmpChannel.Process(largerHeadroomInput)
    assert np.array_equal(
        stableOutputAfterFailure, stableOutputBeforeFailure
    )
    assert np.array_equal(
        stableFeedbackAfterFailure, stableFeedbackBeforeFailure
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
    fixedOutput, fixedFeedbackOutput = fixedChannel.Process(fixedInput)
    for publicOutput in (fixedOutput, fixedFeedbackOutput):
        assert publicOutput.dtype == np.complex128
        assert np.array_equal(publicOutput.real, np.rint(publicOutput.real))
        assert np.array_equal(publicOutput.imag, np.rint(publicOutput.imag))

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
        {"txIqGainImbalanceDb": np.inf},
        {"txIqPhaseImbalanceDegrees": "invalid"},
        {"txDcOffset": complex(np.nan, 0.0)},
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
        {"calibrationDigitalHeadroomDb": -0.1},
        {"calibrationProbeStepDb": 0.0},
        {"calibrationRegularization": 0.0},
    )
    for invalidParameters in invalidConfigurations:
        try:
            Channel(parameters=invalidParameters)
        except (TypeError, ValueError) as error:
            assert "Allowed" in str(error), (
                f"missing allowed range for {invalidParameters!r}: {error}"
            )
        else:
            raise AssertionError(
                f"invalid channel configuration accepted: "
                f"{invalidParameters!r}"
            )


def CheckTwoToneAnalogPowerReporting() -> None:
    """Verify calibrated two-tone power for every public input representation.

    Processing details:
        Algorithm: Generate an unequal-amplitude two-tone record whose active
        RMS maps to 20 dBm under the project's 25 dBm normalized full-scale
        convention. Compare direct and facade dictionary results for floating
        NumPy/list inputs, repeat with 16-bit public I/Q codes while omitting
        width to exercise the raw-record default, and add long leading,
        trailing, and internal idle intervals to prove inactive samples do not
        dilute the reported active PA output power. Finally verify that every
        order-specific IM3/IM5/IM7 result carries the same analog-power metric.

    Returns:
        result: None. Assertion failures identify fixed-point decoding, active
            power, result-shape, or per-order facade regressions.
    """

    sampleRateHz = 100.0e6
    toneFrequenciesHz = (-2.0e6, 2.0e6)
    maximumOutputPowerDbm = 25.0
    expectedOutputPowerDbm = 20.0
    expectedNormalizedRms = float(
        np.power(
            10.0,
            (expectedOutputPowerDbm - maximumOutputPowerDbm) / 20.0,
        )
    )
    commonGeneratorParameters = {
        "sampleRateHz": sampleRateHz,
        "toneFrequenciesHz": toneFrequenciesHz,
        "toneAmplitudes": (1.0, 0.6),
        "tonePhasesDegrees": (17.0, -31.0),
        "numSamples": 4096,
        "rmsLevel": expectedNormalizedRms,
    }
    expectedMetricNames = {
        "fundamentalLowerDbfs",
        "fundamentalUpperDbfs",
        "fundamentalAverageDbfs",
        "im3LowerDbc",
        "im3UpperDbc",
        "im3WorstDbc",
        "im5LowerDbc",
        "im5UpperDbc",
        "im5WorstDbc",
        "im7LowerDbc",
        "im7UpperDbc",
        "im7WorstDbc",
        "worstIntermodulationDbc",
        "outputPowerDbm",
    }
    analysisParameters = {
        "settlingSamples": 0,
        "maximumOutputPowerDbm": maximumOutputPowerDbm,
        "activePowerThresholdDb": -60.0,
        "activeGapToleranceSamples": 16,
    }

    floatingWaveform = WaveGenTwoTone(
        parameters={**commonGeneratorParameters, "width": 0}
    ).Generate()
    directFloatingMetrics = TwoToneAnalysis(
        floatingWaveform,
        parameters={**analysisParameters, "width": 0},
    ).Analyze(floatingWaveform.samples)
    assert isinstance(directFloatingMetrics, dict)
    assert set(directFloatingMetrics) == expectedMetricNames
    assert np.isclose(
        directFloatingMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.01,
    )
    for invalidFullScale in (True, np.bool_(True), "2.0"):
        try:
            TwoToneAnalysis(
                floatingWaveform,
                outputFullScaleAmplitude=invalidFullScale,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                "TwoToneAnalysis accepted invalid output full scale "
                f"{invalidFullScale!r}"
            )

    # Raw normalized floating records remain backward compatible when width is
    # omitted, while an explicit zero remains the authoritative declaration.
    inferredFloatingMetrics = Analysis.AnalyzeTwoTone(
        floatingWaveform.samples,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters=analysisParameters,
    )
    floatingNumpyMetrics = Analysis.AnalyzeTwoTone(
        floatingWaveform.samples,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters=analysisParameters,
        width=0,
    )
    floatingListMetrics = Analysis.AnalyzeTwoTone(
        floatingWaveform.samples.tolist(),
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=list(toneFrequenciesHz),
        parameters=analysisParameters,
        width=0,
    )
    for floatingMetrics in (
        inferredFloatingMetrics,
        floatingNumpyMetrics,
        floatingListMetrics,
    ):
        assert isinstance(floatingMetrics, dict)
        assert set(floatingMetrics) == expectedMetricNames
        assert np.isclose(
            floatingMetrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=0.01,
        )

    fixedWaveform = WaveGenTwoTone(
        parameters={**commonGeneratorParameters, "width": 16}
    ).Generate()
    directFixedMetrics = TwoToneAnalysis(
        fixedWaveform,
        parameters={**analysisParameters, "width": 16},
    ).Analyze(fixedWaveform.samples)
    assert isinstance(directFixedMetrics, dict)
    assert set(directFixedMetrics) == expectedMetricNames
    assert np.isclose(
        directFixedMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )

    expandedOutputScale = 2.0
    expandedOutputCodes = FixedPoint(
        16, expandedOutputScale
    ).EncodeComplex(floatingWaveform.samples)
    expandedOutputMetrics = TwoToneAnalysis(
        floatingWaveform,
        parameters={
            **analysisParameters,
            "width": 16,
            "outputFullScaleAmplitude": expandedOutputScale,
        },
    ).Analyze(expandedOutputCodes)
    assert np.isclose(
        expandedOutputMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )
    automaticExpandedOutputMetrics = TwoToneAnalysis(
        floatingWaveform,
        parameters={**analysisParameters, "width": 16},
    ).Analyze(expandedOutputCodes)
    assert np.isclose(
        automaticExpandedOutputMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )
    assert np.isclose(
        expandedOutputMetrics["fundamentalAverageDbfs"],
        directFixedMetrics["fundamentalAverageDbfs"]
        - 20.0 * np.log10(expandedOutputScale),
        atol=0.02,
    )
    assert expandedOutputMetrics["im3WorstDbc"] < -85.0
    expandedRawMetadata = Analysis.BuildTwoToneWaveform(
        expandedOutputCodes,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        width=16,
        outputFullScaleAmplitude=expandedOutputScale,
    )
    assert np.isclose(
        expandedRawMetadata.rmsLevel,
        expectedNormalizedRms,
        atol=2.0e-4,
    )
    automaticExpandedRawMetadata = Analysis.BuildTwoToneWaveform(
        expandedOutputCodes,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        width=16,
    )
    assert np.isclose(
        automaticExpandedRawMetadata.rmsLevel,
        expectedNormalizedRms,
        atol=2.0e-4,
    )
    expandedRawMetrics = Analysis.AnalyzeTwoTone(
        expandedOutputCodes,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters={
            **analysisParameters,
            "width": 16,
            "outputFullScaleAmplitude": expandedOutputScale,
        },
    )
    assert np.isclose(
        expandedRawMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )
    automaticExpandedRawMetrics = Analysis.AnalyzeTwoTone(
        expandedOutputCodes,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters={**analysisParameters, "width": 16},
    )
    assert np.isclose(
        automaticExpandedRawMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )

    # Transmit metadata and receiver sample width describe different physical
    # boundaries. A floating reference may therefore accompany a 16-bit VSA
    # capture without changing its frequency metadata or power calibration.
    mixedBoundaryMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples,
        floatingWaveform,
        parameters=analysisParameters,
        width=16,
    )
    assert np.isclose(
        mixedBoundaryMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )
    inferredMixedBoundaryMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples,
        floatingWaveform,
        parameters=analysisParameters,
    )
    directInferredMixedBoundaryMetrics = TwoToneAnalysis(
        floatingWaveform,
        parameters=analysisParameters,
    ).Analyze(fixedWaveform.samples)
    for inferredMixedMetrics in (
        inferredMixedBoundaryMetrics,
        directInferredMixedBoundaryMetrics,
    ):
        assert np.isclose(
            inferredMixedMetrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=0.02,
        )
    mixedRawBoundaryMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples,
        floatingWaveform.samples,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters=analysisParameters,
    )
    assert np.isclose(
        mixedRawBoundaryMetrics["outputPowerDbm"],
        expectedOutputPowerDbm,
        atol=0.02,
    )

    # Integer-code-shaped raw NumPy/list records outside normalized range are
    # recognized as signed 16-bit I/Q when width is omitted. Treating these
    # codes as floating amplitudes produced the former 112 dBm failure, so the
    # near-20 dBm comparison is also a direct scale regression.
    fixedNumpyMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters=analysisParameters,
    )
    fixedListMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples.tolist(),
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=list(toneFrequenciesHz),
        parameters=analysisParameters,
    )
    for fixedMetrics in (
        fixedNumpyMetrics,
        fixedListMetrics,
    ):
        assert isinstance(fixedMetrics, dict)
        assert set(fixedMetrics) == expectedMetricNames
        assert np.isclose(
            fixedMetrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=0.02,
        )
        assert fixedMetrics["outputPowerDbm"] <= maximumOutputPowerDbm
        for absoluteMetricName in (
            "fundamentalLowerDbfs",
            "fundamentalUpperDbfs",
            "fundamentalAverageDbfs",
        ):
            assert np.isclose(
                fixedMetrics[absoluteMetricName],
                directFixedMetrics[absoluteMetricName],
                atol=1.0e-12,
            )

    idleSampleCount = 768
    paddedFloatingSignal = np.concatenate(
        (
            np.zeros(384, dtype=np.complex128),
            floatingWaveform.samples,
            np.zeros(idleSampleCount, dtype=np.complex128),
            floatingWaveform.samples,
            np.zeros(512, dtype=np.complex128),
        )
    )
    paddedFloatingMetrics = Analysis.AnalyzeTwoTone(
        paddedFloatingSignal,
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=toneFrequenciesHz,
        parameters=analysisParameters,
        width=0,
    )
    paddedFixedSignal = FixedPoint(16).EncodeComplex(
        paddedFloatingSignal
    )
    paddedFixedMetrics = Analysis.AnalyzeTwoTone(
        paddedFixedSignal.tolist(),
        sampleRateHz=sampleRateHz,
        toneFrequenciesHz=list(toneFrequenciesHz),
        parameters=analysisParameters,
    )
    for paddedMetrics, toleranceDb in (
        (paddedFloatingMetrics, 0.01),
        (paddedFixedMetrics, 0.02),
    ):
        assert isinstance(paddedMetrics, dict)
        assert np.isclose(
            paddedMetrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=toleranceDb,
        )

    for orderMethod in (
        Analysis.CalculateIm3,
        Analysis.CalculateIm5,
        Analysis.CalculateIm7,
    ):
        orderMetrics = orderMethod(
            fixedWaveform.samples.tolist(),
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=list(toneFrequenciesHz),
            parameters=analysisParameters,
        )
        assert isinstance(orderMetrics, dict)
        assert "outputPowerDbm" in orderMetrics
        assert np.isclose(
            orderMetrics["outputPowerDbm"],
            expectedOutputPowerDbm,
            atol=0.02,
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
    assert -42.5 < baselineMetrics["im3WorstDbc"] < -39.5

    facadeMetrics = Analysis.AnalyzeTwoTone(
        baselineOutput,
        floatingWaveform,
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    for metricName, metricValue in baselineMetrics.items():
        assert np.isclose(facadeMetrics[metricName], metricValue)

    rawNumpyMetrics = Analysis.AnalyzeTwoTone(
        baselineOutput,
        floatingWaveform.samples,
        sampleRateHz=floatingWaveform.sampleRateHz,
        toneFrequenciesHz=floatingWaveform.toneFrequenciesHz,
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    rawListMetrics = Analysis.AnalyzeTwoTone(
        baselineOutput.tolist(),
        floatingWaveform.samples.tolist(),
        sampleRateHz=floatingWaveform.sampleRateHz,
        toneFrequenciesHz=list(floatingWaveform.toneFrequenciesHz),
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    standaloneRawMetrics = Analysis.AnalyzeTwoTone(
        baselineOutput.tolist(),
        sampleRateHz=floatingWaveform.sampleRateHz,
        toneFrequenciesHz=floatingWaveform.toneFrequenciesHz,
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    for rawMetrics in (
        rawNumpyMetrics,
        rawListMetrics,
        standaloneRawMetrics,
    ):
        for metricName, metricValue in baselineMetrics.items():
            assert np.isclose(rawMetrics[metricName], metricValue)
        for metricName in (
            "im3LowerDbc",
            "im3UpperDbc",
            "im5LowerDbc",
            "im5UpperDbc",
            "im7LowerDbc",
            "im7UpperDbc",
        ):
            assert metricName in rawMetrics

    rawIm3 = Analysis.CalculateIm3(
        baselineOutput.tolist(),
        sampleRateHz=floatingWaveform.sampleRateHz,
        toneFrequenciesHz=list(floatingWaveform.toneFrequenciesHz),
        parameters={
            "settlingSamples": 64,
            "width": 0,
        },
    )
    assert np.isclose(rawIm3["worstDbc"], baselineMetrics["im3WorstDbc"])
    fixedRawMetrics = Analysis.AnalyzeTwoTone(
        fixedWaveform.samples.tolist(),
        sampleRateHz=fixedWaveform.sampleRateHz,
        toneFrequenciesHz=fixedWaveform.toneFrequenciesHz,
        parameters={
            "settlingSamples": 64,
            "width": 16,
        },
    )
    assert np.isfinite(fixedRawMetrics["im3WorstDbc"])
    try:
        Analysis.AnalyzeTwoTone(
            baselineOutput.tolist(),
            parameters={"settlingSamples": 64, "width": 0},
        )
    except ValueError as error:
        assert "sampleRateHz and toneFrequenciesHz" in str(error)
    else:
        raise AssertionError("raw two-tone input must require physical metadata")

    orderMethods = (
        (3, Analysis.CalculateIm3),
        (5, Analysis.CalculateIm5),
        (7, Analysis.CalculateIm7),
    )
    for nonlinearOrder, orderMethod in orderMethods:
        orderMetrics = orderMethod(
            baselineOutput,
            floatingWaveform,
            parameters={
                "settlingSamples": 64,
                "width": 0,
            },
        )
        expectedPrefix = f"im{nonlinearOrder}"
        expectedFrequenciesHz = (
            floatingWaveform.IntermodulationFrequencies(nonlinearOrder)
        )
        assert set(orderMetrics) == {
            "nonlinearOrder",
            "lowerFrequencyHz",
            "upperFrequencyHz",
            "lowerProductDbfs",
            "upperProductDbfs",
            "lowerDbc",
            "upperDbc",
            "worstDbc",
            "outputPowerDbm",
        }
        assert orderMetrics["nonlinearOrder"] == nonlinearOrder
        assert np.isclose(
            orderMetrics["lowerFrequencyHz"],
            expectedFrequenciesHz[0],
        )
        assert np.isclose(
            orderMetrics["upperFrequencyHz"],
            expectedFrequenciesHz[1],
        )
        assert np.isclose(
            orderMetrics["lowerDbc"],
            baselineMetrics[f"{expectedPrefix}LowerDbc"],
        )
        assert np.isclose(
            orderMetrics["upperDbc"],
            baselineMetrics[f"{expectedPrefix}UpperDbc"],
        )
        assert np.isclose(
            orderMetrics["worstDbc"],
            baselineMetrics[f"{expectedPrefix}WorstDbc"],
        )
        assert np.isclose(
            orderMetrics["lowerProductDbfs"],
            baselineMetrics["fundamentalLowerDbfs"]
            + baselineMetrics[f"{expectedPrefix}LowerDbc"],
        )
        assert np.isclose(
            orderMetrics["upperProductDbfs"],
            baselineMetrics["fundamentalUpperDbfs"]
            + baselineMetrics[f"{expectedPrefix}UpperDbc"],
        )
        assert np.isclose(
            orderMetrics["outputPowerDbm"],
            baselineMetrics["outputPowerDbm"],
        )

    try:
        Analysis.CalculateIntermodulationOrder(
            baselineOutput,
            floatingWaveform,
            9,
            parameters={"settlingSamples": 64, "width": 0},
        )
    except ValueError as error:
        assert "3, 5, or 7" in str(error)
    else:
        raise AssertionError("unsupported IM order must be rejected")

    unequalWaveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": 100.0e6,
            "toneFrequenciesHz": (-2.0e6, 2.0e6),
            "toneAmplitudes": (1.0, 0.5),
            "numSamples": 4096,
            "rmsLevel": 0.25,
            "width": 0,
        }
    ).Generate()
    unequalCubicOutput = unequalWaveform.samples + (
        0.01
        * unequalWaveform.samples
        * np.abs(unequalWaveform.samples) ** 2
    )
    unequalIm3 = Analysis.CalculateIm3(
        unequalCubicOutput,
        unequalWaveform,
        parameters={"settlingSamples": 0, "width": 0},
    )
    expectedProductImbalanceDb = 20.0 * np.log10(2.0)
    assert np.isclose(
        unequalIm3["lowerProductDbfs"]
        - unequalIm3["upperProductDbfs"],
        expectedProductImbalanceDb,
        atol=0.1,
    )
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
                width=16,
                numSamples=4096,
                numIterations=2,
                outputDirectory=outputDirectory,
            )
        )
        assert len(benchmarkRows) == 8
        assert all(
            abs(row.metrics["outputPowerDbm"] - 20.0) <= 0.25
            for row in benchmarkRows
        )
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
        Algorithm: Run a compact Rapp/Wiener/GMP/Doherty frequency and tone-spacing
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
        assert len(result.frequencyResponse) == 24
        assert len(result.memoryEffect) == 12
        assert len(result.powerSweep) == 8
        assert len(result.summaries) == 4
        assert len(result.recommendations) == 20
        resultDocument = result.ToDict()
        assert len(resultDocument["powerSweep"]) == 8
        assert len(resultDocument["recommendations"]) == 20
        assert tuple(
            summary.modelName for summary in result.summaries
        ) == ("rapp", "wiener", "gmp", "doherty")
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
        gmpPowerPoints = tuple(
            powerPoint
            for powerPoint in result.powerSweep
            if powerPoint.modelName == "gmp"
        )
        dohertyPowerPoints = tuple(
            powerPoint
            for powerPoint in result.powerSweep
            if powerPoint.modelName == "doherty"
        )
        gmpIm3Dbc = np.asarray(
            [powerPoint.im3WorstDbc for powerPoint in gmpPowerPoints],
            dtype=float,
        )
        dohertyIm3Dbc = np.asarray(
            [powerPoint.im3WorstDbc for powerPoint in dohertyPowerPoints],
            dtype=float,
        )
        assert np.all(np.diff(gmpIm3Dbc) > 4.0)
        assert np.all(np.diff(dohertyIm3Dbc) > 4.0)
        assert -42.5 < gmpIm3Dbc[-1] < -40.0
        assert dohertyIm3Dbc[-1] < -28.0
        rappSummary = next(
            summary
            for summary in result.summaries
            if summary.modelName == "rapp"
        )
        assert rappSummary.gainRippleDb < 1.0e-6
        assert abs(rappSummary.groupDelayNs) < 1.0e-6
        assert rappSummary.phaseNonlinearityDegrees < 1.0e-6
        assert rappSummary.im3SpacingVariationDb < 0.10
        assert rappSummary.maximumIm3AsymmetryDb < 0.10
        assert rappSummary.dynamicGainHysteresisDb < 0.10
        assert rappSummary.dynamicPhaseHysteresisDegrees < 0.10
        for modelName in ("rapp", "wiener", "gmp", "doherty"):
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
        assert len(recommendationDocument["recommendations"]) == 20
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
        assert (
            dpdGmpDocument["configuration"]["seed"]
            != dpdGmpDocument["configuration"]["validationSeed"]
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


def CheckDpdLmsModelAndBenchmark() -> None:
    """Verify sample updates, commit modes, synchronization, and tracking.

    Processing details:
        Algorithm: Compare causal sample-built GMP rows against the batch
        feature matrix, recover a known complex linear-plus-cubic mapping with
        frame-commit NLMS, prove that shadow coefficients change before active
        coefficients, exercise strict LMS sample commit and running scales,
        accept a longer delayed indirect feedback capture, preserve fixed-point
        public codes and unknown-key warnings, then run the compact
        stationary/drift benchmark and require its artifacts and documented
        implementation distinctions.

    Returns:
        result: None. Assertions identify adaptive DPD regressions.
    """

    randomGenerator = np.random.default_rng(733)
    referenceSignal = (
        randomGenerator.standard_normal(4096)
        + 1j * randomGenerator.standard_normal(4096)
    )
    referenceSignal *= 0.25 / np.sqrt(
        np.mean(np.abs(referenceSignal) ** 2)
    )
    targetSignal = (
        (1.03 + 0.01j) * referenceSignal
        + (0.18 - 0.04j)
        * referenceSignal
        * np.abs(referenceSignal) ** 2
    )
    featureProbe = DpdLms(
        parameters={
            "nonlinearOrders": (1, 3, 5),
            "memoryDepth": 3,
            "crossMemoryDepth": 2,
            "width": 0,
        }
    )
    probeReference = referenceSignal[:128]
    batchFeatureMatrix = featureProbe.BuildBasisChunk(
        probeReference,
        0,
        probeReference.size,
    )
    featureProbe.BeginFrame(probeReference)
    sampleFeatureMatrix = np.vstack(
        tuple(
            featureProbe.BuildFeatureVector(
                complex(referenceSample)
            )
            for referenceSample in probeReference
        )
    )
    assert np.allclose(
        sampleFeatureMatrix,
        batchFeatureMatrix,
        atol=1.0e-14,
    )

    frameDpd = DpdLms(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "learningRate": 0.10,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "leakageFactor": 0.0,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    activeBeforeSample = frameDpd.GetCoefficients()
    frameDpd.BeginFrame(referenceSignal)
    frameDpd.UpdateSample(
        complex(referenceSignal[0]),
        complex(targetSignal[0]),
    )
    assert np.array_equal(
        frameDpd.GetCoefficients(),
        activeBeforeSample,
    )
    assert not np.array_equal(
        frameDpd.GetAdaptiveCoefficients(),
        activeBeforeSample,
    )
    frameDpd.ResetCoefficients()
    trainingResult = frameDpd.UpdateFromLabels(
        referenceSignal,
        targetSignal,
    )
    assert trainingResult.sampleCount == referenceSignal.size
    assert trainingResult.updateCount == referenceSignal.size
    assert trainingResult.afterNmseDb < -100.0
    assert trainingResult.afterNmseDb < (
        trainingResult.beforeNmseDb - 40.0
    )
    assert trainingResult.coefficientsCommitted is True
    assert frameDpd.GetLastLmsTrainingResult() == trainingResult
    recoveredCoefficients = frameDpd.GetCoefficients()
    assert np.allclose(
        recoveredCoefficients,
        np.asarray((1.03 + 0.01j, 0.18 - 0.04j)),
        atol=2.0e-4,
    )

    sampleDpd = DpdLms(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "adaptationMode": "lms",
            "learningRate": 0.001,
            "featureScaleMode": "running",
            "coefficientCommitMode": "sample",
            "leakageFactor": 0.0,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    sampleActiveBefore = sampleDpd.GetCoefficients()
    sampleDpd.BeginFrame()
    sampleDpd.UpdateSample(
        complex(referenceSignal[0]),
        complex(targetSignal[0]),
    )
    assert not np.array_equal(
        sampleDpd.GetCoefficients(),
        sampleActiveBefore,
    )

    indirectDpd = DpdLms(
        parameters={
            "nonlinearOrders": (1,),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "learningRate": 0.05,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "leakageFactor": 0.0,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    delayedFeedback = np.concatenate(
        (
            np.zeros(17, dtype=np.complex128),
            referenceSignal,
            np.zeros(11, dtype=np.complex128),
        )
    )
    indirectResult = indirectDpd.UpdateIndirect(
        referenceSignal,
        delayedFeedback,
        sampleRateHz=80.0e6,
    )
    assert indirectResult.sampleCount == referenceSignal.size
    assert indirectResult.afterNmseDb < -100.0

    fixedIndirectPhysical = referenceSignal[:1024]
    fixedIndirectInput = FixedPoint(16).EncodeComplex(
        fixedIndirectPhysical
    )
    fixedIndirectOutput = FixedPoint(16, 2.0).EncodeComplex(
        fixedIndirectPhysical
    )
    disabledGainCompensation = {
        "enableComplexGainCompensation": False,
    }
    fixedIndirectGmp = DpdGmp(
        parameters={
            "nonlinearOrders": (1,),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "ridgeFactor": 1.0e-9,
            "maximumOutputMagnitude": None,
            "width": 16,
        }
    )
    fixedIndirectGmp.FitIndirect(
        fixedIndirectInput,
        fixedIndirectOutput,
        sampleRateHz=80.0e6,
        signalProcessingParameters=disabledGainCompensation,
        paOutputFullScaleAmplitude=2.0,
    )
    assert abs(fixedIndirectGmp.GetCoefficients()[0] - 1.0) < 0.01

    fixedIndirectLms = DpdLms(
        parameters={
            "nonlinearOrders": (1,),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "learningRate": 0.05,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "leakageFactor": 0.0,
            "maximumOutputMagnitude": None,
            "width": 16,
        }
    )
    fixedIndirectLms.UpdateIndirect(
        fixedIndirectInput,
        fixedIndirectOutput,
        sampleRateHz=80.0e6,
        signalProcessingParameters=disabledGainCompensation,
        paOutputFullScaleAmplitude=2.0,
    )
    assert abs(fixedIndirectLms.GetCoefficients()[0] - 1.0) < 0.01

    fixedFormat = FixedPoint(16)
    fixedReference = fixedFormat.EncodeComplex(referenceSignal)
    fixedTarget = fixedFormat.EncodeComplex(targetSignal)
    fixedDpd = DpdLms(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "learningRate": 0.10,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "leakageFactor": 0.0,
            "maximumOutputMagnitude": None,
            "width": 16,
        }
    )
    fixedDpd.UpdateFromLabels(fixedReference, fixedTarget)
    fixedOutput = fixedDpd.Process(fixedReference)
    assert fixedOutput.dtype == np.complex128
    assert np.array_equal(fixedOutput.real, np.rint(fixedOutput.real))
    assert np.array_equal(fixedOutput.imag, np.rint(fixedOutput.imag))

    with warnings.catch_warnings(record=True) as warningRecords:
        warnings.simplefilter("always")
        warnedDpd = DpdLms(
            parameters={
                "width": 0,
                "unknownLmsOption": 5,
            }
        )
    assert warnedDpd.width == 0
    assert any(
        "unknownLmsOption" in str(warningRecord.message)
        for warningRecord in warningRecords
    )

    from tests.BenchMark import (
        DpdLmsBenchmarkConfig,
        RunDpdLmsBenchmark,
    )

    with TemporaryDirectory() as temporaryDirectory:
        outputDirectory = Path(temporaryDirectory)
        benchmarkResult = RunDpdLmsBenchmark(
            DpdLmsBenchmarkConfig(
                numSamples=1024,
                seed=739,
                outputDirectory=outputDirectory,
            )
        )
        assert benchmarkResult.updateCountPerFrame == 1024
        assert benchmarkResult.trackingImprovementDb > 20.0
        assert (
            benchmarkResult.lmsAfterTrackingNmseDb
            < benchmarkResult.lmsBeforeTrackingNmseDb
        )
        assert (
            outputDirectory / "dpd_lms_benchmark.csv"
        ).exists()
        assert (
            outputDirectory / "dpd_lms_benchmark.json"
        ).exists()

    for documentName, requiredText in (
        ("DPD-LMS.md", "逐样点更新"),
        ("DpdLms.md", "SmallestLMS.py"),
        ("BenchMark.md", "DPD-LMS逐样点"),
    ):
        documentText = (
            GetProjectRoot() / "doc" / documentName
        ).read_text(encoding="utf-8")
        assert requiredText in documentText


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

    from tests.BenchMark import (
        DpdGmpBenchmarkConfig,
        EvaluateDpdGmpWifiStage,
        GenerateDpdGmpIlcLabel,
        ParseBenchmarkArguments,
    )

    try:
        DpdGmpBenchmarkConfig(seed=701, validationSeed=701).Validate()
    except ValueError as error:
        assert "distinct integers" in str(error)
    else:
        raise AssertionError("DPD-GMP benchmark accepted its training seed")

    with patch.object(
        sys,
        "argv",
        [
            "BenchMark.py",
            "--dpd-gmp",
            "--seed",
            "701",
            "--validation-seed",
            "809",
        ],
    ):
        parsedDpdGmpConfig = ParseBenchmarkArguments()
    assert isinstance(parsedDpdGmpConfig, DpdGmpBenchmarkConfig)
    assert parsedDpdGmpConfig.seed == 701
    assert parsedDpdGmpConfig.validationSeed == 809

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

    augmentedTarget = (
        1.08 * referenceSignal
        + 0.16 * referenceSignal * np.abs(referenceSignal) ** 2
        + 0.12 * np.conj(referenceSignal)
        + 0.05
        * np.conj(referenceSignal)
        * np.abs(referenceSignal) ** 2
    )
    directOnlyDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "ridgeFactor": 1.0e-8,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    augmentedDpd = AugmentedDpdGmp(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "ridgeFactor": 1.0e-8,
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )
    directOnlyResult = directOnlyDpd.Fit(
        referenceSignal,
        augmentedTarget,
    )
    augmentedResult = augmentedDpd.Fit(
        referenceSignal,
        augmentedTarget,
    )
    assert augmentedResult.featureCount == 2 * directOnlyResult.featureCount
    assert augmentedResult.afterNmseDb < directOnlyResult.afterNmseDb - 20.0
    assert np.linalg.norm(
        augmentedDpd.GetImageCoefficients()
    ) > 0.01

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
    fixedAugmentedDpd = AugmentedDpdGmp(
        parameters={"width": 16}
    )
    assert np.array_equal(
        fixedAugmentedDpd.Process(fixedWaveform.samples),
        fixedWaveform.samples,
    )
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

    # The deployable ordinary and piecewise GMP DPDs must not collapse every
    # noiseless operating point onto the power-independent first-order memory
    # floor. Train each point on one frame, validate on another, and require
    # the PA-only long cubic envelope-memory tail to leave a visible residual
    # that grows with conducted output power.
    powerValidationConfig = DpdGmpBenchmarkConfig(
        mcs=5,
        numDataSymbols=2,
        seed=91,
        validationSeed=809,
        trainingPowerDbm=(1.0, 16.0, 20.0),
        optimizedOutputPowerDbm=16.0,
        stressOutputPowerDbm=20.0,
        numIterations=8,
        width=0,
    )

    (
        powerTrainingWaveform,
        powerValidationWaveform,
    ) = tuple(
        WaveGenWifi(
            parameters={
                "frameFormat": "EHT",
                "bandwidthMhz": 20,
                "sampleRateHz": 80.0e6,
                "mcs": 5,
                "numDataSymbols": 2,
                "seed": seed,
                "width": 0,
            }
        ).Generate()
        for seed in (91, 809)
    )
    dpdPowerEvmRows = []
    baselinePowerEvmDb = None
    for dpdType in (DpdGmp, PiecewiseDpdGmp):
        correctedEvmDbValues = []
        currentBaselineEvmDbValues = []
        for outputPowerDbm in (1.0, 16.0, 20.0):
            powerPaModel = PaModel(
                parameters={"modelName": "gmp", "width": 0}
            )
            powerReference, learnedInput = GenerateDpdGmpIlcLabel(
                powerValidationConfig,
                powerTrainingWaveform,
                powerPaModel,
                outputPowerDbm,
            )
            powerDpd = dpdType(parameters={"width": 0})
            powerDpd.FitFromIlc(powerReference, learnedInput)
            baselineMetrics = EvaluateDpdGmpWifiStage(
                powerValidationConfig,
                powerValidationWaveform,
                powerPaModel,
                outputPowerDbm,
            )
            correctedMetrics = EvaluateDpdGmpWifiStage(
                powerValidationConfig,
                powerValidationWaveform,
                powerPaModel,
                outputPowerDbm,
                powerDpd,
            )
            currentBaselineEvmDbValues.append(
                float(baselineMetrics["evmDb"])
            )
            correctedEvmDbValues.append(
                float(correctedMetrics["evmDb"])
            )
            assert abs(
                float(correctedMetrics["outputPowerDbm"])
                - outputPowerDbm
            ) <= 0.15
            assert (
                float(correctedMetrics["evmDb"])
                < float(baselineMetrics["evmDb"]) - 4.0
            )
        currentBaselineEvmDb = np.asarray(
            currentBaselineEvmDbValues,
            dtype=float,
        )
        correctedEvmDb = np.asarray(correctedEvmDbValues, dtype=float)
        if baselinePowerEvmDb is None:
            baselinePowerEvmDb = currentBaselineEvmDb
        else:
            assert np.allclose(
                currentBaselineEvmDb,
                baselinePowerEvmDb,
                rtol=0.0,
                atol=1.0e-10,
            )
        assert np.all(np.diff(correctedEvmDb) > 3.0)
        assert correctedEvmDb[-1] - correctedEvmDb[0] > 8.0
        dpdPowerEvmRows.append(correctedEvmDb)
    assert baselinePowerEvmDb is not None
    assert np.all(
        np.abs(
            baselinePowerEvmDb
            - np.asarray((-51.6, -40.7, -33.0), dtype=float)
        )
        <= 1.0
    )
    correctedPowerEvmDb = np.vstack(dpdPowerEvmRows)
    assert np.all(np.isfinite(correctedPowerEvmDb))
    expectedCorrectedPowerEvmDb = np.asarray(
        (
            (-57.4, -46.5, -38.0),
            (-57.4, -46.5, -38.0),
        ),
        dtype=float,
    )
    correctedPowerTolerancesDb = np.asarray(
        (1.0, 1.0, 0.75), dtype=float
    )
    assert np.all(
        np.abs(
            correctedPowerEvmDb - expectedCorrectedPowerEvmDb
        )
        <= correctedPowerTolerancesDb
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


def CheckChannelAnalysisAndCoupledDpd() -> None:
    """Verify channel extraction, causal inversion, and benchmark improvements.

    Processing details:
        Algorithm: Measure a known delayed complex leakage path, compare its
        gain/phase/delay with configured truth, verify measured causal
        pre-coupling inversion numerically, run the compact nonlinear MIMO
        benchmark, require every declared before/after trend, and check all
        documented CSV/JSON/PNG artifacts.

    Returns:
        result: None. Assertions expose channel or coupled-DPD regressions.
    """

    measurementChannel = Channel(
        parameters={
            "prePaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": -20.0,
                    "phaseDegrees": 30.0,
                    "integerDelaySamples": 2,
                },
            ),
            "width": 0,
        }
    )
    channelAnalyzer = ChannelAnalyse(
        parameters={
            "sampleRateHz": 80.0e6,
            "channelBandwidthHz": 20.0e6,
            "fftLength": 512,
            "impulseLength": 16,
            "width": 0,
        }
    )
    preMeasurement = channelAnalyzer.Measure(
        measurementChannel.ApplyPrePaCoupling,
        2,
        "pre-PA",
    )
    measuredLeakage = preMeasurement.GetPath(0, 1)
    assert measuredLeakage.detected
    assert abs(measuredLeakage.gainDb + 20.0) < 0.01
    assert abs(measuredLeakage.phaseDegrees - 30.0) < 0.01
    assert abs(measuredLeakage.groupDelaySamples - 2.0) < 0.01
    assert measuredLeakage.flatnessDb < 0.01
    assert preMeasurement.worstConditionNumber > 1.0

    fixedMeasurementPa = PaModel(
        modelName="gmp",
        width=8,
        outputFullScaleAmplitude=4.0,
    )
    fixedMeasurementChannel = Channel(
        paModel=fixedMeasurementPa,
        parameters={
            "sampleMode": "forward",
            "width": 16,
            "outputFullScaleAmplitude": 2.0,
        },
    )
    fixedChannelAnalyzer = ChannelAnalyse(
        parameters={
            "sampleRateHz": 80.0e6,
            "channelBandwidthHz": 20.0e6,
            "fftLength": 128,
            "impulseLength": 8,
            "probeDelaySamples": 2,
            "width": 16,
        }
    )
    fixedPostPaMeasurement = fixedChannelAnalyzer.Measure(
        fixedMeasurementChannel.ProcessPaOutput,
        1,
        "fixed post-PA",
    )
    fixedDirectPath = fixedPostPaMeasurement.GetPath(0, 0)
    assert abs(fixedDirectPath.gainDb) < 0.01
    assert fixedDirectPath.flatnessDb < 0.01

    identityModels = (
        DpdGmp(parameters={"width": 0}),
        DpdGmp(parameters={"width": 0}),
    )
    coupledDpd = CouplingAwareDpdGmp(
        identityModels,
        preMeasurement,
        None,
        parameters={"width": 0},
    )
    randomGenerator = np.random.default_rng(809)
    desiredPaInput = 0.1 * (
        randomGenerator.standard_normal((512, 2))
        + 1j * randomGenerator.standard_normal((512, 2))
    )
    rawDacInput = coupledDpd.BuildDacInput(
        desiredPaInput
    )
    restoredPaInput = measurementChannel.ApplyPrePaCoupling(
        rawDacInput
    )
    inversionNmseDb = 10.0 * np.log10(
        np.mean(np.abs(restoredPaInput - desiredPaInput) ** 2)
        / np.mean(np.abs(desiredPaInput) ** 2)
    )
    assert inversionNmseDb < -120.0

    with warnings.catch_warnings(record=True) as warningRecords:
        warnings.simplefilter("always")
        warnedAnalyzer = ChannelAnalyse(
            parameters={
                "width": 0,
                "unknownChannelAnalysisOption": 5,
            }
        )
    assert warnedAnalyzer.width == 0
    assert any(
        "unknownChannelAnalysisOption"
        in str(warningRecord.message)
        for warningRecord in warningRecords
    )

    from tests.BenchMark import (
        ChannelAnalysisBenchmarkConfig,
        RunChannelAnalysisBenchmark,
    )

    with TemporaryDirectory() as temporaryDirectory:
        outputDirectory = Path(temporaryDirectory)
        result = RunChannelAnalysisBenchmark(
            ChannelAnalysisBenchmarkConfig(
                numDataSymbols=2,
                numIterations=8,
                outputDirectory=outputDirectory,
            )
        )
        assert len(result.prePaMeasurement.paths) == 4
        assert len(result.postPaMeasurement.paths) == 4
        assert len(result.stages) == 4
        assert len(result.improvements) == 4
        assert len(result.iqImbalanceStages) == 15
        conventionalIrr = min(
            stage.irrDb
            for stage in result.iqImbalanceStages
            if stage.methodName == "Conventional GMP"
        )
        augmentedIrr = max(
            stage.irrDb
            for stage in result.iqImbalanceStages
            if stage.methodName == "Augmented GMP"
        )
        assert augmentedIrr < conventionalIrr - 40.0
        assert all(
            improvement.expectationMet
            for improvement in result.improvements
        )
        assert (
            result.stages[-1].evmDb
            < result.stages[1].evmDb
        )
        assert (
            result.stages[-1].normalizedMseDb
            < result.stages[1].normalizedMseDb
        )
        for artifactName in (
            "channel_analysis.json",
            "channel_path_measurements.csv",
            "channel_frequency_response.csv",
            "channel_dpd_comparison.csv",
            "channel_dpd_improvements.csv",
            "channel_analysis.png",
            "iq_gmp_comparison.csv",
            "iq_gmp_comparison.png",
        ):
            assert (outputDirectory / artifactName).exists()
        savedResult = json.loads(
            (
                outputDirectory / "channel_analysis.json"
            ).read_text(encoding="utf-8")
        )
        assert all(
            improvement["expectationMet"]
            for improvement in savedResult["improvements"]
        )
    channelDocument = (
        GetProjectRoot() / "doc" / "ChannelAnalyse.md"
    ).read_text(encoding="utf-8")
    for requiredText in (
        "平坦度",
        "耦合参数",
        "群时延",
        "CouplingAwareDpdGmp",
        "修改前后性能比较",
        "channel_analysis.png",
    ):
        assert requiredText in channelDocument


def BuildNaiveGmpOutput(
    paModel: GMPPA,
    inputSignal: np.ndarray,
) -> np.ndarray:
    """Evaluate GMP terms without sharing delayed or envelope arrays.

    Processing details:
        Algorithm: Recreate the original direct implementation by visiting
        main, lagging, and leading coefficient dictionaries in their stored
        order and calling ``DelaySignal`` independently for every term. This
        deliberately slow oracle detects any numerical or causal change in
        the optimized production implementation without using elapsed time.

    Args:
        paModel: Configured GMP PA whose coefficient dictionaries are read.
        inputSignal: Finite one-dimensional complex test waveform.

    Returns:
        result: Direct per-term GMP output in complex128 precision.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128)
    outputSignal = np.zeros_like(complexInput)
    for (
        nonlinearOrder,
        memoryIndex,
    ), coefficient in paModel.mainCoefficients.items():
        delayedSignal = DelaySignal(complexInput, memoryIndex)
        outputSignal += (
            coefficient
            * delayedSignal
            * np.abs(delayedSignal) ** (nonlinearOrder - 1)
        )
    for (
        nonlinearOrder,
        memoryIndex,
        crossIndex,
    ), coefficient in paModel.laggingCoefficients.items():
        carrierSignal = DelaySignal(complexInput, memoryIndex)
        envelopeSignal = DelaySignal(
            complexInput,
            memoryIndex + crossIndex,
        )
        outputSignal += (
            coefficient
            * carrierSignal
            * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
        )
    for (
        nonlinearOrder,
        memoryIndex,
        crossIndex,
    ), coefficient in paModel.leadingCoefficients.items():
        carrierSignal = DelaySignal(
            complexInput,
            memoryIndex + crossIndex,
        )
        envelopeSignal = DelaySignal(complexInput, memoryIndex)
        outputSignal += (
            coefficient
            * carrierSignal
            * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
        )
    return outputSignal


def EstimateLegacyIntegerDelay(
    signalProcessor: SigProc,
    measuredSignal: np.ndarray,
) -> int:
    """Reproduce the former scalar normalized-correlation lag search.

    Processing details:
        Algorithm: Calculate the same FFT linear correlation used by
        ``SigProc``, then visit candidate lags one by one, slice each overlap,
        calculate its two energies, and preserve the first-maximum tie rule.
        This scalar oracle verifies the vectorized prefix-sum implementation.

    Args:
        signalProcessor: Processor containing the reference and search bound.
        measuredSignal: Padded or cropped finite complex measurement.

    Returns:
        result: Signed measured-versus-reference integer delay in samples.
    """

    complexMeasured = signalProcessor.ValidateSignal(
        measuredSignal,
        "measuredSignal",
    )
    referenceSignal = signalProcessor.referenceSignal
    referenceLength = referenceSignal.size
    measuredLength = complexMeasured.size
    fullLength = referenceLength + measuredLength - 1
    fftLength = 1 << int(np.ceil(np.log2(max(fullLength, 2))))
    correlation = np.fft.ifft(
        np.fft.fft(complexMeasured, fftLength)
        * np.fft.fft(np.conj(referenceSignal[::-1]), fftLength)
    )[:fullLength]
    lags = np.arange(fullLength, dtype=int) - (referenceLength - 1)
    maximumDelay = signalProcessor.ResolveMaximumIntegerDelay()
    minimumOverlap = max(
        16,
        min(referenceLength, measuredLength) // 4,
    )
    bestDelay = 0
    bestScore = -np.inf
    for correlationIndex, candidateLag in enumerate(lags):
        if abs(int(candidateLag)) > maximumDelay:
            continue
        referenceStart = max(0, -int(candidateLag))
        referenceStop = min(
            referenceLength,
            measuredLength - int(candidateLag),
        )
        if referenceStop - referenceStart < minimumOverlap:
            continue
        measuredStart = referenceStart + int(candidateLag)
        measuredStop = measuredStart + (
            referenceStop - referenceStart
        )
        referenceEnergy = np.sum(
            np.abs(referenceSignal[referenceStart:referenceStop]) ** 2
        )
        measuredEnergy = np.sum(
            np.abs(complexMeasured[measuredStart:measuredStop]) ** 2
        )
        normalization = np.sqrt(
            max(
                referenceEnergy * measuredEnergy,
                np.finfo(float).tiny,
            )
        )
        candidateScore = float(
            np.abs(correlation[correlationIndex]) / normalization
        )
        if candidateScore > bestScore:
            bestScore = candidateScore
            bestDelay = int(candidateLag)
    if not np.isfinite(bestScore):
        raise RuntimeError("unable to estimate legacy integer delay")
    return bestDelay


def FindLegacyActiveSampleMask(
    powerCalibration: PowerCalibration,
    inputSignal: np.ndarray,
) -> np.ndarray:
    """Recreate adjacent-active-index gap filling for active detection.

    Processing details:
        Algorithm: Threshold every chain relative to its peak and revisit
        each adjacent pair of active indices, filling an internal inactive
        interval only when its sample count does not exceed the configured
        tolerance. This is the pre-vectorization behavior used as an oracle.

    Args:
        powerCalibration: Calibrator providing threshold and gap parameters.
        inputSignal: Finite complex vector or samples-by-chain matrix.

    Returns:
        result: Legacy boolean activity mask with input-matching orientation.
    """

    complexSignal = np.asarray(inputSignal, dtype=np.complex128)
    inputWasVector = complexSignal.ndim == 1
    signalMatrix = (
        complexSignal.reshape(-1, 1)
        if inputWasVector
        else complexSignal
    )
    instantaneousPower = np.abs(signalMatrix) ** 2
    peakPowerPerChain = np.max(instantaneousPower, axis=0)
    relativePowerThreshold = 10.0 ** (
        float(powerCalibration.parameters["activePowerThresholdDb"])
        / 10.0
    )
    activeMask = (
        instantaneousPower
        > peakPowerPerChain.reshape(1, -1)
        * relativePowerThreshold
    )
    gapTolerance = int(
        powerCalibration.parameters["activeGapToleranceSamples"]
    )
    if gapTolerance > 0:
        for chainIndex in range(signalMatrix.shape[1]):
            activeIndices = np.flatnonzero(activeMask[:, chainIndex])
            for activePairIndex in range(activeIndices.size - 1):
                gapStart = int(activeIndices[activePairIndex]) + 1
                gapStop = int(activeIndices[activePairIndex + 1])
                if gapStop - gapStart <= gapTolerance:
                    activeMask[gapStart:gapStop, chainIndex] = True
    return activeMask[:, 0] if inputWasVector else activeMask


def CheckPerformanceOptimizationEquivalence() -> None:
    """Verify accelerated hot paths without fragile wall-clock assertions.

    Processing details:
        Algorithm: Compare cached GMP, vectorized integer-delay, and run-based
        activity detection against direct legacy oracles; count MIMO Analysis
        demodulation and periodogram calls; exercise short-probe assisted
        overlap for positive, negative, and cropped records; and round-trip
        the descriptor LDPC code through clean and noisy soft observations.

    Returns:
        result: None. Structural counts and exact numerical assertions expose
            semantic regressions independently of host or CI execution speed.
    """

    randomGenerator = np.random.default_rng(20260825)

    # Cached delayed signals and envelope powers must retain bit-for-bit GMP
    # behavior for built-in coefficients, sparse measured coefficients, and
    # input records shorter than every configured causal delay.
    defaultGmp = GMPPA()
    defaultInput = 0.17 * (
        randomGenerator.normal(size=257)
        + 1j * randomGenerator.normal(size=257)
    )
    assert np.array_equal(
        defaultGmp.Process(defaultInput),
        BuildNaiveGmpOutput(defaultGmp, defaultInput),
    )
    customGmp = GMPPA(
        GMPConfig(
            nonlinearOrders=(1, 3, 5),
            memoryDepth=4,
            crossMemoryDepth=3,
            mainCoefficients={
                (1, 0): 1.07 + 0.02j,
                (3, 2): -0.13 + 0.04j,
                (5, 3): 0.018 - 0.006j,
            },
            laggingCoefficients={
                (3, 1, 2): 0.031 - 0.009j,
                (5, 0, 3): -0.004 + 0.002j,
            },
            leadingCoefficients={
                (3, 0, 2): -0.022 + 0.007j,
                (5, 1, 3): 0.003 - 0.001j,
            },
        )
    )
    customInput = 0.21 * (
        randomGenerator.normal(size=113)
        + 1j * randomGenerator.normal(size=113)
    )
    for testedInput in (
        customInput,
        customInput[:1],
        customInput[:2],
        customInput[:3],
    ):
        assert np.array_equal(
            customGmp.Process(testedInput),
            BuildNaiveGmpOutput(customGmp, testedInput),
        )

    # Channel validates one live configuration transaction at its public
    # boundary. Nested ideal stages may share intermediate arrays, while the
    # public input and both returned arrays retain independent ownership.
    channelInput = 0.13 * (
        randomGenerator.normal(size=257)
        + 1j * randomGenerator.normal(size=257)
    )
    acceleratedChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={"sampleMode": "forward", "width": 0},
    )
    validationDepths = []
    finiteProofNames = []
    originalValidateParameters = acceleratedChannel.ValidateParameters
    originalPrepareSignal = acceleratedChannel.PrepareSignal

    def RecordChannelValidation() -> None:
        """Record whether a Channel validation is outer or nested.

        Processing details:
            Algorithm: Save the trusted-transaction depth immediately before
            delegating to the real validator without changing its result.

        Returns:
            result: None. The captured depths are asserted below.
        """

        validationDepths.append(
            acceleratedChannel._trustedProcessingDepth
        )
        originalValidateParameters()

    def RecordSignalPreparation(
        inputSignal: np.ndarray,
        signalName: str,
        forceValidation: bool = False,
    ) -> np.ndarray:
        """Record full finite-value proofs around the optimized Channel.

        Processing details:
            Algorithm: Count outer-boundary and explicitly forced checks, then
            delegate shape, type, and finite-value handling to PrepareSignal.

        Args:
            inputSignal: Candidate complex vector or matrix.
            signalName: Diagnostic name supplied by the production caller.
            forceValidation: Whether an external result requires a new proof.

        Returns:
            result: The production PrepareSignal result.
        """

        if (
            forceValidation
            or acceleratedChannel._trustedProcessingDepth == 0
        ):
            finiteProofNames.append(signalName)
        return originalPrepareSignal(
            inputSignal,
            signalName,
            forceValidation,
        )

    with patch.object(
        acceleratedChannel,
        "ValidateParameters",
        new=RecordChannelValidation,
    ), patch.object(
        acceleratedChannel,
        "PrepareSignal",
        new=RecordSignalPreparation,
    ):
        channelOutput, feedbackOutput = acceleratedChannel.Process(
            channelInput
        )
    assert validationDepths.count(0) == 1
    assert finiteProofNames.count("inputSignal") == 1
    assert finiteProofNames.count("paOutputSignal") == 1
    assert np.array_equal(channelOutput, feedbackOutput)
    assert not np.shares_memory(channelOutput, feedbackOutput)
    assert not np.shares_memory(channelOutput, channelInput)
    assert acceleratedChannel._trustedProcessingDepth == 0

    # Every standalone identity helper preserves its historical defensive-copy
    # contract even though the same stages alias intermediates inside Process.
    identityHelperOutputs = (
        acceleratedChannel.ApplyMimoCoupling(
            channelInput, "prePaCouplingPaths"
        ),
        acceleratedChannel.ApplyIqImbalanceStage(
            channelInput,
            0.0,
            0.0,
            0.0 + 0.0j,
            "testIqStage",
            None,
            None,
        ),
        acceleratedChannel.ApplyTransmitterIqImbalance(channelInput),
        acceleratedChannel.ApplyPhaseRotation(channelInput),
        acceleratedChannel.AddNoise(channelInput),
        acceleratedChannel.ApplyFeedbackLinearResponse(channelInput),
        acceleratedChannel.ApplyFeedbackNonlinearity(channelInput),
        acceleratedChannel.ApplyFeedbackTimingAndFrequency(channelInput),
        acceleratedChannel.ApplyFeedbackIqImbalance(channelInput),
        acceleratedChannel.ApplyFeedbackAdc(channelInput),
        acceleratedChannel.ApplyCalibrationDrive(channelInput),
    )
    for identityHelperOutput in identityHelperOutputs:
        assert np.array_equal(identityHelperOutput, channelInput)
        assert not np.shares_memory(identityHelperOutput, channelInput)

    # A callback returning nonfinite data and a finite PA result that overflows
    # during post-coupling must both fail and restore the transaction depth.
    failingPaModel = PaModel(modelName="wiener", width=0)
    failingChannel = Channel(
        paModel=failingPaModel,
        parameters={"sampleMode": "forward", "width": 0},
    )
    protectedChannelInput = channelInput.copy()

    def MutatePaInputAndReturnNan(
        inputSignal: np.ndarray,
        **thermalParameters: object,
    ) -> np.ndarray:
        """Mutate a received PA buffer and return an invalid waveform.

        Processing details:
            Algorithm: Overwrite the array supplied across the third-party PA
            boundary, ignore scheduling keywords, and return NaNs so both
            caller-input ownership and exception cleanup are exercised.

        Args:
            inputSignal: Floating waveform supplied to the synthetic PA.
            thermalParameters: Channel thermal scheduling keyword values.

        Returns:
            result: Same-shape nonfinite waveform rejected by Channel.
        """

        inputSignal[...] = 0.0 + 0.0j
        return np.full(inputSignal.shape, np.nan + 0.0j)

    with patch.object(
        failingPaModel,
        "ProcessThermalPeriodFloating",
        side_effect=MutatePaInputAndReturnNan,
    ):
        try:
            failingChannel.ProcessOutputPathsFloating(channelInput)
        except ValueError as error:
            assert "paOutputSignal" in str(error)
        else:
            raise AssertionError("Channel accepted a nonfinite PA output")
        try:
            failingChannel.Process(channelInput)
        except ValueError as error:
            assert "paOutputSignal" in str(error)
        else:
            raise AssertionError(
                "Channel.Process accepted a nonfinite PA output"
            )
    assert np.array_equal(channelInput, protectedChannelInput)
    assert failingChannel._trustedProcessingDepth == 0
    recoveredOutput, recoveredFeedback = (
        failingChannel.ProcessOutputPathsFloating(channelInput)
    )
    assert np.all(np.isfinite(recoveredOutput))
    assert np.array_equal(recoveredOutput, recoveredFeedback)

    overflowPaModel = MimoPaModel(
        parameters={"numTransmitChains": 2, "width": 0}
    )
    overflowChannel = Channel(
        paModel=overflowPaModel,
        parameters={
            "sampleMode": "forward",
            "postPaCouplingPaths": (
                {
                    "sourceChain": 0,
                    "destinationChain": 1,
                    "gainDb": 20.0 * np.log10(2.0),
                },
            ),
            "width": 0,
        },
    )
    overflowInput = np.column_stack((channelInput, channelInput))
    finiteHugeOutput = np.full(
        overflowInput.shape,
        np.finfo(float).max / 1.5 + 0.0j,
        dtype=np.complex128,
    )
    with patch.object(
        overflowPaModel,
        "ProcessThermalPeriodFloating",
        return_value=finiteHugeOutput,
    ), np.errstate(over="ignore", invalid="ignore"):
        try:
            overflowChannel.ProcessOutputPathsFloating(overflowInput)
        except ValueError as error:
            assert "numeric range" in str(error)
        else:
            raise AssertionError("Channel accepted post-coupling overflow")
    assert overflowChannel._trustedProcessingDepth == 0

    # Caller-owned ChainMap values remain live between public calls; no
    # optimized result or validated configuration crosses that boundary.
    liveChannelParameters = {"phaseDegrees": 0, "width": 0}
    liveChannel = Channel(parameters=liveChannelParameters)
    liveBaseOutput = liveChannel.ProcessPaOutput(channelInput)
    liveChannelParameters["phaseDegrees"] = 90
    liveRotatedOutput = liveChannel.ProcessPaOutput(channelInput)
    assert np.allclose(
        liveRotatedOutput,
        1j * liveBaseOutput,
        rtol=0.0,
        atol=1.0e-15,
    )
    liveChannelParameters["phaseDegrees"] = 45
    try:
        liveChannel.ProcessPaOutput(channelInput)
    except ValueError as error:
        assert "phaseDegrees" in str(error)
    else:
        raise AssertionError("live invalid Channel parameters were cached")

    # A fixed drive-aware calibration encodes its drive-independent DAC preset
    # once per transaction. Each probe receives a fresh copy, and every exit
    # clears the strong input reference held by the transaction-local cache.
    fixedPresetInput = FixedPoint(16).EncodeComplex(
        np.r_[
            np.zeros(8, dtype=np.complex128),
            channelInput,
            np.zeros(8, dtype=np.complex128),
        ]
    )
    fixedPresetChannel = Channel(
        paModel=PaModel(modelName="wiener", width=0),
        parameters={
            "maximumOutputPowerDbm": 25.0,
            "calibrationToleranceDb": 0.05,
            "maximumCalibrationIterations": 60,
            "width": 16,
        },
    )
    originalCalibrationDrive = fixedPresetChannel.ProcessCalibrationDrive
    receivedPresetInputs = []
    encodeInvocationCount = [0]
    originalEncodeComplex = FixedPoint.EncodeComplex

    def MutatingCalibrationDrive(
        inputSignal: np.ndarray,
        driveDbPerChain: object,
    ) -> np.ndarray:
        """Emulate a plant that overwrites each supplied trial buffer.

        Processing details:
            Algorithm: Capture the untouched public codes, evaluate the real
            Channel calibration path, then overwrite only the caller-owned
            trial array to prove the cached source remains isolated.

        Args:
            inputSignal: Public fixed-point trial codes.
            driveDbPerChain: Candidate analog drive values for all chains.

        Returns:
            result: Clean PA output produced before mutating the trial buffer.
        """

        receivedPresetInputs.append(inputSignal.copy())
        calibrationOutput = originalCalibrationDrive(
            inputSignal,
            driveDbPerChain,
        )
        inputSignal[...] = 0.0 + 0.0j
        return calibrationOutput

    def CountEncodeComplex(
        self: FixedPoint,
        inputSignal: np.ndarray,
    ) -> np.ndarray:
        """Count fixed-point encodes while retaining production behavior.

        Processing details:
            Algorithm: Increment a local structural counter and delegate the
            actual rounding, saturation, and copying to the original method.

        Args:
            inputSignal: Normalized complex samples to encode.

        Returns:
            result: Production fixed-point public codes.
        """

        encodeInvocationCount[0] += 1
        return originalEncodeComplex(self, inputSignal)

    fixedPresetChannel.ProcessCalibrationDrive = MutatingCalibrationDrive
    with patch.object(
        FixedPoint,
        "EncodeComplex",
        new=CountEncodeComplex,
    ):
        fixedPresetChannel.CalibratePaInput(fixedPresetInput, 20.0)
    fixedPresetCalibration = fixedPresetChannel._powerCalibration
    if fixedPresetCalibration is None:
        raise AssertionError("Channel did not retain its calibrator")
    fixedPresetMetrics = (
        fixedPresetCalibration.GetLastCalibrationMetrics()
    )
    fixedPresetIterationCount = int(fixedPresetMetrics["iterationCount"])
    assert fixedPresetIterationCount > 1
    assert encodeInvocationCount[0] == fixedPresetIterationCount + 1
    assert len(receivedPresetInputs) == fixedPresetIterationCount
    for receivedPresetInput in receivedPresetInputs[1:]:
        assert np.array_equal(
            receivedPresetInput,
            receivedPresetInputs[0],
        )
    assert fixedPresetCalibration._activeFixedDrivePresetCache is None
    assert fixedPresetCalibration._electricalCalibrationTransactionActive is False

    secondPresetInput = np.roll(
        np.conjugate(fixedPresetInput),
        5,
    )
    secondReceivedStart = len(receivedPresetInputs)
    secondEncodeStart = encodeInvocationCount[0]
    with patch.object(
        FixedPoint,
        "EncodeComplex",
        new=CountEncodeComplex,
    ):
        fixedPresetChannel.CalibratePaInput(secondPresetInput, 20.0)
    secondPresetMetrics = (
        fixedPresetCalibration.GetLastCalibrationMetrics()
    )
    secondPresetIterationCount = int(
        secondPresetMetrics["iterationCount"]
    )
    assert (
        encodeInvocationCount[0] - secondEncodeStart
        == secondPresetIterationCount + 1
    )
    assert len(receivedPresetInputs) == (
        secondReceivedStart + secondPresetIterationCount
    )
    assert not np.array_equal(
        receivedPresetInputs[secondReceivedStart],
        receivedPresetInputs[0],
    )
    assert fixedPresetCalibration._activeFixedDrivePresetCache is None
    assert fixedPresetCalibration._electricalCalibrationTransactionActive is False

    presetProbeMatrix = FixedPoint(16).DecodeComplex(
        fixedPresetInput
    ).reshape(-1, 1)
    presetProbeDriveDb = np.asarray((-5.0,), dtype=float)
    originalThresholdDb = float(
        fixedPresetCalibration.GetParameters()["activePowerThresholdDb"]
    )
    originalGapTolerance = int(
        fixedPresetCalibration.GetParameters()[
            "activeGapToleranceSamples"
        ]
    )
    cacheKeyEncodeStart = encodeInvocationCount[0]
    fixedPresetCalibration._electricalCalibrationTransactionActive = True
    try:
        with patch.object(
            FixedPoint,
            "EncodeComplex",
            new=CountEncodeComplex,
        ):
            fixedPresetCalibration.PrepareDrivePreset(
                presetProbeMatrix,
                presetProbeDriveDb,
                FixedPoint(16),
            )
            fixedPresetCalibration.PrepareDrivePreset(
                presetProbeMatrix,
                presetProbeDriveDb,
                FixedPoint(16),
            )
            assert encodeInvocationCount[0] == cacheKeyEncodeStart + 1
            fixedPresetCalibration.UpdateParameters(
                activePowerThresholdDb=-50.0
            )
            fixedPresetCalibration.PrepareDrivePreset(
                presetProbeMatrix,
                presetProbeDriveDb,
                FixedPoint(16),
            )
            assert encodeInvocationCount[0] == cacheKeyEncodeStart + 2
            fixedPresetCalibration.UpdateParameters(
                activeGapToleranceSamples=0
            )
            fixedPresetCalibration.PrepareDrivePreset(
                presetProbeMatrix,
                presetProbeDriveDb,
                FixedPoint(16),
            )
            assert encodeInvocationCount[0] == cacheKeyEncodeStart + 3
    finally:
        fixedPresetCalibration._activeFixedDrivePresetCache = None
        fixedPresetCalibration._electricalCalibrationTransactionActive = False
        fixedPresetCalibration.UpdateParameters(
            activePowerThresholdDb=originalThresholdDb,
            activeGapToleranceSamples=originalGapTolerance,
        )

    with patch.object(
        fixedPresetCalibration,
        "_paCalibrationProcessMethod",
        side_effect=RuntimeError("synthetic calibration failure"),
    ):
        try:
            fixedPresetCalibration.Calibrate(fixedPresetInput)
        except RuntimeError as error:
            assert "synthetic calibration failure" in str(error)
        else:
            raise AssertionError("synthetic calibration failure was hidden")
    assert fixedPresetCalibration._activeFixedDrivePresetCache is None
    assert fixedPresetCalibration._electricalCalibrationTransactionActive is False

    # Range-tree energy vectorization must select exactly the lag selected by
    # the old scalar overlap-energy loop under padding, cropping, and gain.
    integerReference = (
        randomGenerator.normal(size=1021)
        + 1j * randomGenerator.normal(size=1021)
    )
    integerProcessor = SigProc(
        integerReference,
        40.0e6,
        parameters={"maxIntegerDelaySamples": 96},
    )
    integerGain = 0.64 * np.exp(1j * 0.47)
    integerCases = (
        (
            np.r_[
                np.zeros(31, dtype=np.complex128),
                integerGain * integerReference,
                np.zeros(19, dtype=np.complex128),
            ],
            31,
        ),
        (integerGain * integerReference[27:], -27),
        (integerGain * integerReference[:-53], 0),
    )
    for measuredSignal, expectedDelay in integerCases:
        optimizedDelay = integerProcessor.EstimateIntegerDelay(
            measuredSignal
        )
        legacyDelay = EstimateLegacyIntegerDelay(
            integerProcessor,
            measuredSignal,
        )
        assert optimizedDelay == legacyDelay == expectedDelay

    # Run-boundary gap filling must preserve every boolean decision from the
    # former adjacent-active-index implementation for vectors and matrices.
    activitySignal = np.zeros((128, 3), dtype=np.complex128)
    activitySignal[9:31, 0] = np.exp(
        1j * np.arange(22, dtype=float) * 0.17
    )
    activitySignal[34:61, 0] = 0.8
    activitySignal[79:111, 0] = 0.6j
    activitySignal[3:17, 1] = 0.7 - 0.2j
    activitySignal[18:52, 1] = 0.9 + 0.1j
    activitySignal[68:125, 1] = 0.5
    activitySignal[22:49, 2] = 0.6 + 0.4j
    activitySignal[51:55, 2] = 0.8
    activitySignal[72:91, 2] = 1.0j
    for gapTolerance in (0, 1, 2, 4, 16):
        activeCalibration = PowerCalibration(
            width=0,
            parameters={
                "activePowerThresholdDb": -40.0,
                "activeGapToleranceSamples": gapTolerance,
            },
        )
        for testedSignal in (
            activitySignal[:, 0],
            activitySignal,
        ):
            assert np.array_equal(
                activeCalibration.FindActiveSampleMask(testedSignal),
                FindLegacyActiveSampleMask(
                    activeCalibration,
                    testedSignal,
                ),
            )

    # A 2x2 Analysis call needs one measured and one reference demodulation.
    # Both are rebuilt on reuse so public reference edits cannot leave stale
    # data, while aggregate and per-stream EVM still share the two local grids.
    # Its ACLR path performs one periodogram per physical chain and no more.
    mimoWaveform = WaveGenWifi(
        frameFormat="EHT",
        bandwidthMhz=20,
        mcs=3,
        numDataSymbols=2,
        sampleRateHz=80.0e6,
        numTransmitAntennas=2,
        numSpatialStreams=2,
        spatialMapping="dft",
        seed=621,
        width=0,
    ).Generate()
    mimoMeasurement = (
        0.86 * np.exp(1j * 0.29) * mimoWaveform.samples
    )
    mimoAnalysis = Analysis(
        mimoWaveform.samples,
        mimoWaveform,
        parameters={"maxSegmentLength": 2048},
        width=0,
    )
    if mimoAnalysis.frameProcessor is None:
        raise AssertionError("MIMO Wi-Fi analysis requires FrameProcess")
    originalDemodulate = (
        mimoAnalysis.frameProcessor.DemodulatePreparedWifiData
    )
    with patch.object(
        mimoAnalysis.frameProcessor,
        "DemodulatePreparedWifiData",
        wraps=originalDemodulate,
    ) as demodulateMock, patch(
        "inc.lib.Analysis.AveragePeriodogram",
        wraps=AveragePeriodogram,
    ) as periodogramMock:
        firstMetrics = mimoAnalysis.Analyze(mimoMeasurement)
        firstDemodulationCount = demodulateMock.call_count
        firstPeriodogramCount = periodogramMock.call_count
        assert firstDemodulationCount == 2
        assert firstPeriodogramCount == 2
        secondMetrics = mimoAnalysis.Analyze(mimoMeasurement)
        assert demodulateMock.call_count == firstDemodulationCount + 2
        assert periodogramMock.call_count == firstPeriodogramCount + 2
        assert np.allclose(
            np.asarray(tuple(firstMetrics.values())),
            np.asarray(tuple(secondMetrics.values())),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    originalReference = mimoAnalysis.referenceSignal.copy()
    mimoAnalysis.referenceSignal *= 0.5
    assert (
        mimoAnalysis.CalculatePreparedEvmAlignedMse(
            mimoAnalysis.referenceSignal
        )
        < 1.0e-24
    )
    mimoAnalysis.referenceSignal = originalReference

    # The assisted correlation probe is intentionally shorter than every
    # record. Positive lag, negative lag, and tail-cropped capture geometry
    # must still map exact common sample starts.
    overlapReference = (
        randomGenerator.normal(size=513)
        + 1j * randomGenerator.normal(size=513)
    )
    overlapGain = 0.73 * np.exp(1j * 0.42)
    overlapCases = (
        (
            np.r_[
                np.zeros(29, dtype=np.complex128),
                overlapGain * overlapReference,
                np.zeros(17, dtype=np.complex128),
            ],
            29,
            0,
            513,
        ),
        (
            overlapGain * overlapReference[41:],
            0,
            41,
            472,
        ),
        (
            np.r_[
                np.zeros(11, dtype=np.complex128),
                overlapGain * overlapReference[:-31],
            ],
            11,
            0,
            482,
        ),
    )
    for (
        measuredSignal,
        expectedMeasuredStart,
        expectedReferenceStart,
        expectedOverlapLength,
    ) in overlapCases:
        overlapResult = SigProc.EstimateSignalOverlap(
            measuredSignal,
            overlapReference,
            maximumMeasuredOffsetSamples=80,
            maximumProbeLength=47,
            minimumConfidence=0.99,
        )
        assert overlapResult.receivedStartSample == expectedMeasuredStart
        assert overlapResult.referenceStartSample == expectedReferenceStart
        assert overlapResult.overlapLength == expectedOverlapLength
        assert overlapResult.confidence > 0.999999

    # Range-tree energies must not select a tiny noise-only tail merely
    # because subtracting two large cumulative energies loses precision.
    dynamicGenerator = np.random.default_rng(0)
    dynamicReference = (
        dynamicGenerator.normal(size=128)
        + 1j * dynamicGenerator.normal(size=128)
    )
    dynamicActive = (
        0.8 * dynamicReference
        + dynamicGenerator.normal(size=128)
        + 1j * dynamicGenerator.normal(size=128)
    )
    dynamicMeasured = np.r_[
        dynamicActive,
        1.0e-7
        * (
            dynamicGenerator.normal(size=256)
            + 1j * dynamicGenerator.normal(size=256)
        ),
    ]
    dynamicResult = SigProc.EstimateSignalOverlap(
        dynamicMeasured,
        dynamicReference,
        maximumMeasuredOffsetSamples=200,
        maximumProbeLength=32,
        minimumConfidence=0.0,
    )
    assert dynamicResult.receivedStartSample == 0

    # Clean and noisy soft observations must both recover the original
    # systematic LDPC descriptor payload without relying on an optional
    # version-specific external codec.
    descriptorMessage = randomGenerator.integers(
        0,
        2,
        size=55,
        dtype=np.uint8,
    )
    descriptorCodeword = EncodeDescriptorLdpc(descriptorMessage)
    cleanSoftCodeword = (
        1.0 - 2.0 * descriptorCodeword.astype(float)
    )
    assert np.array_equal(
        DecodeDescriptorLdpc(cleanSoftCodeword),
        descriptorMessage,
    )
    noisySoftCodeword = (
        3.0 * cleanSoftCodeword
        + randomGenerator.normal(scale=0.35, size=90)
    )
    assert np.array_equal(
        DecodeDescriptorLdpc(noisySoftCodeword),
        descriptorMessage,
    )

    # The protocol-layout cache stores immutable bytes. Public calls return
    # distinct read-only views whose write flag cannot be re-enabled, so one
    # caller cannot corrupt a later blind-analysis descriptor search.
    firstLayout = DescriptorLdpcPhysicalLayout()
    secondLayout = DescriptorLdpcPhysicalLayout()
    for firstArray, secondArray in zip(firstLayout, secondLayout):
        assert firstArray is not secondArray
        assert not firstArray.flags.writeable
        assert np.array_equal(firstArray, secondArray)
        try:
            firstArray.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "descriptor layout views must remain read-only"
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
    CheckPerformanceOptimizationEquivalence()
    CheckNoGlobalDataVariables()
    CheckModuleResponsibilityBoundaries()
    CheckBenchmarkSeparation()
    CheckFunctionPrincipleCoverage()
    CheckDocumentationMathCompatibility()
    CheckDocumentationImageLinks()
    CheckDocumentationApiConsistency()
    CheckInternalDefaultConfiguration()
    CheckUnknownConfigurationWarnings()
    CheckFixedPointInterfaces()
    CheckPaThermalModel()
    CheckThermalDisableAndCalibrationBypass()
    CheckChannelPeriodicThermalModes()
    CheckPeriodicThermalEdgeCases()
    CheckFeedbackIqPhasePairCalibration()
    CheckFrequencySelectiveIqImbalance()
    CheckChannelIqEnableControls()
    CheckChannelDualOutputContract()
    CheckChannelModel()
    CheckWifiFormats()
    CheckWifiBandwidths()
    CheckWifiFixedPointHeadroom()
    CheckWifiSpectralMaskAnalysis()
    CheckSampleRateConfiguration()
    CheckMimoSpatialStructure()
    CheckMimoPaAndDpd()
    CheckFormatSpecificMcsValidation()
    CheckIdealMetrics()
    CheckSignalProcessingCompensation()
    CheckPowerEvmCurve()
    CheckIlcPowerOperatingPoints()
    CheckMainIlcPowerPointSelection()
    CheckGuardIntervals()
    CheckRappPaModel()
    CheckGmpPaModel()
    CheckPiecewiseGmpModels()
    CheckDohertyPaModel()
    CheckIlcImprovement()
    CheckIlcFeedbackSynchronization()
    CheckReceiveOnlyWifiAnalysis()
    CheckMseEvmConvergence()
    CheckTwoToneAnalogPowerReporting()
    CheckTwoToneIlcAnalysis()
    CheckPaCharacterizationBenchmark()
    CheckDpdLmsModelAndBenchmark()
    CheckDpdGmpModelAndBenchmark()
    CheckChannelAnalysisAndCoupledDpd()
    print("All DPD-ILC project checks passed.")


if __name__ == "__main__":
    RunTests()
