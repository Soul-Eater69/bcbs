import importlib.metadata as metadata

from idp_eval import (
    CoverageEvaluator,
    FaithfulnessEvaluator,
    EvaluationFramework,
)

# =========================================================
# 1. PRINT PACKAGE VERSIONS
# =========================================================

print("PACKAGE VERSIONS")
print("-" * 80)
print("openai:", metadata.version("openai"))
print("arize-phoenix-evals:", metadata.version("arize-phoenix-evals"))
print()


# =========================================================
# 2. CREATE A DEBUG FRAMEWORK
#
# IMPORTANT:
# - no new retry logic
# - no retry configuration changes
# - no Excel output
# - no Phoenix result persistence
# - same existing judge
# - stops on first failure so we can inspect it properly
# =========================================================

debug_framework = EvaluationFramework(
    judge=judge,
    evaluators=[
        FaithfulnessEvaluator,
        CoverageEvaluator,
    ],
)


# =========================================================
# 3. HELPER TO PRINT THE FULL EXCEPTION CHAIN
# =========================================================

def inspect_exception(exc):
    print()
    print("=" * 100)
    print("EXCEPTION DEBUG INFORMATION")
    print("=" * 100)

    current = exc
    seen = set()
    depth = 0

    while (
        current is not None
        and id(current) not in seen
        and depth < 10
    ):
        seen.add(id(current))

        print()
        print("-" * 100)
        print(f"EXCEPTION LEVEL {depth}")
        print("-" * 100)

        print(
            "Type:",
            f"{type(current).__module__}.{type(current).__name__}",
        )

        print("Message:", str(current))

        # -------------------------------------------------
        # Common useful attributes from Phoenix/OpenAI/httpx
        # -------------------------------------------------

        attrs = [
            "status_code",
            "request_id",
            "code",
            "type",
            "param",
            "current_rate_tokens_per_sec",
            "initial_rate_tokens_per_sec",
            "enforcement_window_seconds",
        ]

        print()
        print("KNOWN ATTRIBUTES")

        found_attribute = False

        for attr in attrs:
            try:
                value = getattr(current, attr, None)
            except Exception:
                value = None

            if value is not None:
                found_attribute = True
                print(f"  {attr}: {value}")

        if not found_attribute:
            print("  none")


        # -------------------------------------------------
        # Inspect HTTP response if preserved by exception
        # -------------------------------------------------

        response = getattr(current, "response", None)

        if response is not None:
            print()
            print("HTTP RESPONSE")

            try:
                print(
                    "  status_code:",
                    response.status_code,
                )
            except Exception:
                pass

            try:
                headers = response.headers

                print()
                print("IMPORTANT RESPONSE HEADERS")

                interesting_headers = {}

                for key, value in headers.items():
                    lower = key.lower()

                    if (
                        "retry" in lower
                        or "rate" in lower
                        or "request-id" in lower
                        or "request_id" in lower
                        or "x-ms" in lower
                    ):
                        interesting_headers[key] = value

                if interesting_headers:
                    for key, value in interesting_headers.items():
                        print(f"  {key}: {value}")
                else:
                    print("  none found")

            except Exception as header_exc:
                print(
                    "  Could not inspect headers:",
                    type(header_exc).__name__,
                    str(header_exc),
                )

            try:
                body = response.text

                if body:
                    print()
                    print("RESPONSE BODY")
                    print(body[:5000])

            except Exception as body_exc:
                print(
                    "  Could not inspect response body:",
                    type(body_exc).__name__,
                    str(body_exc),
                )


        # -------------------------------------------------
        # Inspect exception __dict__ for anything useful
        # -------------------------------------------------

        try:
            exception_dict = vars(current)

            if exception_dict:
                print()
                print("EXCEPTION __dict__ KEYS")

                safe_keys = []

                for key in exception_dict.keys():
                    lower = key.lower()

                    # Avoid accidentally dumping sensitive content.
                    if not any(
                        secret_word in lower
                        for secret_word in [
                            "token",
                            "secret",
                            "password",
                            "authorization",
                            "credential",
                            "api_key",
                        ]
                    ):
                        safe_keys.append(key)

                print(" ", safe_keys)

        except Exception:
            pass


        # -------------------------------------------------
        # Walk chained exception
        # -------------------------------------------------

        next_exception = current.__cause__

        if next_exception is None:
            next_exception = current.__context__

        current = next_exception
        depth += 1


    print()
    print("=" * 100)
    print("END EXCEPTION DEBUG")
    print("=" * 100)


# =========================================================
# 4. RUN CASES SEQUENTIALLY UNTIL FIRST FAILURE
#
# This intentionally does NOT use evaluate_many().
#
# Reason:
# We want to know exactly which case fails and inspect
# the complete exception immediately.
# =========================================================

print()
print("STARTING RATE-LIMIT DEBUG RUN")
print("=" * 100)
print(f"Total cases: {len(cases)}")
print("Metrics: faithfulness, coverage")
print("=" * 100)
print()


for index, case in enumerate(cases, start=1):

    case_name = (
        case.case_id
        if case.case_id is not None
        else f"index-{index - 1}"
    )

    print(
        f"[{index}/{len(cases)}] "
        f"Running case={case_name}"
    )

    try:
        result = debug_framework.evaluate(
            case,
            metrics=[
                "faithfulness",
                "coverage",
            ],
            run_name="rate_limit_debug",
            dataset_name="golden_set_augmented_tagged.csv",
        )

        faithfulness = result["faithfulness"]
        coverage = result["coverage"]

        print(
            "  ✓ SUCCESS"
            f" | faithfulness={faithfulness.score}"
            f" ({faithfulness.label})"
            f" | coverage={coverage.score}"
            f" ({coverage.label})"
        )

        print()

    except Exception as exc:
        print()
        print(
            f"✗ FAILED"
            f" | case={case_name}"
            f" | position={index}/{len(cases)}"
        )

        inspect_exception(exc)

        print()
        print(
            "DEBUG RUN STOPPED AT FIRST FAILURE."
        )

        break


else:
    print()
    print("=" * 100)
    print("ALL CASES COMPLETED WITHOUT FAILURE")
    print("=" * 100)
