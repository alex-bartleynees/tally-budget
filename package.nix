{ lib, python3Packages, }:
python3Packages.buildPythonPackage {
  pname = "tally";
  version = "0.1.0";
  pyproject = true;

  src = ./.;

  nativeBuildInputs = with python3Packages; [ hatchling ];
  dependencies = with python3Packages; [ pyyaml ];
  nativeCheckInputs = with python3Packages; [ pytestCheckHook ];

  # Skip tests that use subprocess to call 'uv run tally'
  disabledTestPaths = [
    "tests/test_cli.py"
    "tests/test_rule_snapshots.py"
  ];

  meta = {
    description =
      "Let AI classify your transactions - LLM-powered spending categorization";
    homepage = "https://github.com/davidfowl/tally";
    license = lib.licenses.mit;
    mainProgram = "tally";
    maintainers = [ "AlexBN" ];
    platforms = lib.platforms.all;
  };
}
