using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;

namespace Tuna.Generators;

public sealed partial class OperationsGenerator
{
    private const string Differentiation = "global::Tuna.Differentiation";

    private static string? GenerateDifferentiation(
        StringBuilder code, IMethodSymbol method, AttributeData attribute, UserSpec spec, Compilation compilation, string scalar, string access)
    {
        string? derivative = NamedString(attribute, "Derivative");
        string[]? partials = NamedStrings(attribute, "PartialDerivatives");
        if ((derivative is null && partials is null) ||
            compilation.GetTypeByMetadataName("Tuna.Differentiation.ScalarNode") is not { ContainingAssembly.Name: "Tuna.Differentiation" })
        {
            return null;
        }

        string[]? derivativeNames = spec.Inputs.Length == 1
            ? derivative is not null && partials is null ? new[] { derivative } : null
            : derivative is null ? partials : null;
        if (derivativeNames is null || derivativeNames.Length != spec.Inputs.Length)
        {
            return $"[ElementwiseOperation] の {method.Name} では、入力が1つなら Derivative、2・3個なら Over の指定順で PartialDerivatives を指定してください";
        }

        var doubleType = compilation.GetSpecialType(SpecialType.System_Double);
        if (method.ReturnType is ITypeParameterSymbol typeParameter)
        {
            var doubleConstraints = new HashSet<string>(doubleType.AllInterfaces.Select(Display), StringComparer.Ordinal);
            if (typeParameter.HasReferenceTypeConstraint ||
                typeParameter.ConstraintTypes.Any(c => !doubleConstraints.Contains(DisplayWithTypeArgument(c, "double"))))
            {
                return $"[ElementwiseOperation] の {method.Name} のスカラーのノード版では、要素型に double を指定できること";
            }
        }
        else if (method.ReturnType.SpecialType is not (SpecialType.System_Single or SpecialType.System_Double))
        {
            return $"[ElementwiseOperation] の {method.Name} の微分規則には、float・double またはメソッドの型引数を要素型にしてください";
        }

        var over = NamedStrings(attribute, "Over") ?? new[] { spec.Inputs[0].Name };
        var names = new string[spec.Inputs.Length];
        for (int i = 0; i < names.Length; i++)
        {
            string name = derivativeNames[Array.IndexOf(over, spec.Inputs[i].Name)];
            if (!IsIdentifier(name) || !method.ContainingType.GetMembers(name).OfType<IMethodSymbol>().Any(candidate => MatchesDerivative(method, candidate)))
            {
                return $"[ElementwiseOperation] の {method.Name} の微分規則 {name} は、元のメソッドと同じ型の static メソッドにし、元の引数に続けて結果を受け取って要素型を返してください";
            }

            names[i] = name;
        }

        EmitScalarDifferentiation(code, method, spec, names, scalar, access);
        EmitArrayDifferentiation(code, method, spec, names, scalar, access);
        return null;
    }

    private static bool MatchesDerivative(IMethodSymbol method, IMethodSymbol derivative)
    {
        if (!derivative.IsStatic || derivative.Arity != method.Arity || derivative.Parameters.Length != method.Parameters.Length + 1 ||
            derivative.ReturnsByRef || derivative.ReturnsByRefReadonly ||
            derivative.Parameters.Any(p => p.RefKind != RefKind.None))
        {
            return false;
        }

        string element = DisplayWithTypeArgument(method.ReturnType, "T");
        return DisplayWithTypeArgument(derivative.ReturnType, "T") == element &&
            DisplayWithTypeArgument(derivative.Parameters.Last().Type, "T") == element &&
            method.Parameters.Select(p => DisplayWithTypeArgument(p.Type, "T"))
                .SequenceEqual(derivative.Parameters.Take(method.Parameters.Length).Select(p => DisplayWithTypeArgument(p.Type, "T")));
    }

    private static string DisplayWithTypeArgument(ITypeSymbol type, string argument)
        => string.Concat(type.ToDisplayParts(SymbolDisplayFormat.FullyQualifiedFormat)
            .Select(part => part.Kind == SymbolDisplayPartKind.TypeParameterName ? argument : part.ToString()));

    private static void EmitScalarDifferentiation(
        StringBuilder code, IMethodSymbol method, UserSpec spec, string[] derivatives, string scalar, string access)
    {
        string node = Differentiation + ".ScalarNode";
        string container = Display(method.ContainingType);
        bool specialize = method.ReturnType is ITypeParameterSymbol;
        string typeParameters = method.Arity == 0 || specialize ? "" : $"<{method.TypeParameters[0].Name}>";
        string typeArguments = specialize ? "<double>" : typeParameters;
        string constraints = specialize ? "" : Constraints(method.TypeParameters, s => s);
        string e = specialize ? "double" : Display(method.ReturnType);
        string prefix = "__tuna";
        while (method.Parameters.Any(p => p.Name.StartsWith(prefix, StringComparison.Ordinal)))
        {
            prefix += "_";
        }

        var arguments = method.Parameters.Select(p => p.Name).ToArray();
        var inputOrdinals = spec.Inputs.Select(p => p.Ordinal).ToArray();
        string parameters = string.Join(", ", method.Parameters.Select(p =>
        {
            string type = specialize ? DisplayWithTypeArgument(p.Type, "double") : Display(p.Type);
            return $"{(inputOrdinals.Contains(p.Ordinal) ? node : type)} {p.Name}";
        }));
        code.Append($"    /// <summary>Applies {scalar} to a reverse-mode scalar node. <para>{scalar} を逆モード自動微分のスカラーノードに適用します。</para></summary>\n");
        code.Append($"    {access} static {node} {method.Name}{typeParameters}({parameters})\n{constraints}    {{\n");
        foreach (var input in spec.Inputs)
        {
            string local = prefix + input.Ordinal;
            code.Append($"        {e} {local} = ({e}){input.Name}.Value;\n");
            arguments[input.Ordinal] = local;
        }

        string value = prefix + "Value";
        string args = string.Join(", ", arguments);
        code.Append($"        {e} {value} = {container}.{method.Name}{typeArguments}({args});\n");
        string requiresGradient = string.Join(" || ", spec.Inputs.Select(p => p.Name + ".RequiresGradient"));
        code.Append($"        if (!({requiresGradient}))\n        {{\n            return {value};\n        }}\n\n");
        string edges = string.Join(",\n            ", spec.Inputs.Select((p, i) =>
            $"({p.Name}, {p.Name}.RequiresGradient ? (double){container}.{derivatives[i]}{typeArguments}({args}, {value}) : 0)"));
        code.Append($"        return {node}.Record({value},\n        [\n            {edges},\n        ]);\n    }}\n\n");
    }

    private static void EmitArrayDifferentiation(
        StringBuilder code, IMethodSymbol method, UserSpec spec, string[] derivatives, string scalar, string access)
    {
        string e = Display(method.ReturnType);
        string typeArguments = method.Arity == 0 ? "" : $"<{method.TypeParameters[0].Name}>";
        string constraints = Constraints(method.TypeParameters, s => s, differentiate: method.ReturnType is ITypeParameterSymbol);
        string container = Display(method.ContainingType);
        string rule = "__" + (spec.OperatorContainer is null ? "" : spec.OperatorContainer + "_") + spec.OperatorName + "Differentiation";
        string ruleType = rule + typeArguments;
        string inputType = $"{Differentiation}.DifferentiableOperationContext<{e}>";
        string matrix = TypeOf(true, ViewKind.ReadOnly, e);
        var inputs = spec.Inputs.Select(p => p.Ordinal).ToArray();
        var extras = method.Parameters.Where(p => !inputs.Contains(p.Ordinal)).ToArray();
        string state = string.Join(", ", extras.Select(p => p.Name));
        string nodes = string.Join(", ", spec.Inputs.Select(p => p.Name));
        string first = spec.Inputs[0].Name;
        foreach (bool isMatrix in new[] { false, true })
        {
            string node = $"{Differentiation}.{(isMatrix ? "Matrix" : "Vector")}Node<{e}>";
            string parameters = string.Join(", ", method.Parameters.Select(p =>
                $"{(inputs.Contains(p.Ordinal) ? node : Display(p.Type))} {p.Name}"));
            string size = isMatrix ? $"{first}.Rows, {first}.Columns" : $"{first}.Length";
            string kind = isMatrix ? "matrix" : "vector";
            string kindJa = isMatrix ? "行列" : "ベクトル";
            code.Append(Method($"<summary>Applies {scalar} to a reverse-mode {kind} node. <para>{scalar} を逆モード自動微分の{kindJa}ノードに適用します。</para></summary>",
                access, node, method.Name + typeArguments, parameters, constraints,
                $"return {node}.Record(new {ruleType}({state}), [{nodes}], {size});"));
        }

        string constructor = extras.Length == 0 ? "" : "(" + string.Join(", ", extras.Select(p => $"{Display(p.Type)} arg{p.Ordinal}")) + ")";
        code.Append($"    private readonly struct {ruleType}{constructor} : {Differentiation}.IDifferentiableOperation<{e}>\n");
        code.Append(Indent(constraints)).Append("    {\n");
        var forwardArguments = method.Parameters.Select(p => $"arg{p.Ordinal}").ToArray();
        var backwardArguments = (string[])forwardArguments.Clone();
        for (int i = 0; i < inputs.Length; i++)
        {
            forwardArguments[inputs[i]] = $"inputs[{i}]";
            backwardArguments[inputs[i]] = $"input{i}[row, column]";
        }

        string forward = string.Join(", ", forwardArguments);
        code.Append($"        public void Forward({inputType} inputs, {TypeOf(true, ViewKind.Writable, e)} destination)\n");
        code.Append($"            => {container}.{method.Name}{typeArguments}({forward}, destination);\n\n");
        code.Append($"        public void PullBack({inputType} inputs, {matrix} output, {matrix} outputGradient)\n        {{\n");
        for (int i = 0; i < inputs.Length; i++)
        {
            code.Append($"            var input{i} = inputs[{i}];\n");
        }

        string backward = string.Join(", ", backwardArguments);
        for (int i = 0; i < inputs.Length; i++)
        {
            code.Append($@"
            if (inputs.RequiresGradient({i}))
            {{
                var gradient = inputs.GetGradient({i});
                for (int column = 0; column < output.Columns; column++)
                {{
                    for (int row = 0; row < output.Rows; row++)
                    {{
                        var adjoint = outputGradient[row, column];
                        if (adjoint == default)
                        {{
                            continue;
                        }}

                        var derivative = {container}.{derivatives[i]}{typeArguments}({backward}, output[row, column]);
                        if (derivative != default)
                        {{
                            gradient[row, column] += adjoint * derivative;
                        }}
                    }}
                }}
            }}
");
        }

        code.Append("        }\n    }\n\n");
    }
}
