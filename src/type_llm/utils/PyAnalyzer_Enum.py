from enum import Enum


class RefKind(str, Enum):
    #use-ref
    UseKind = "Use"
    CallKind = "Call"
    
    #def-ref
    DefineKind = "Define"
    SetKind = "Set"

    
    #specitial-ref
    InheritKind = "Inherit"
    
    #other-ref
    AliasTo = "Alias"
    ImportKind = "Import"
    ContainKind = "Contain"
    HasambiguousKind = "Hasambiguous"
    Annotate = "Annotate"

Def_RefKinds = [RefKind.DefineKind, RefKind.SetKind]
Use_RefKinds = [RefKind.UseKind, RefKind.CallKind]
Ignored_RefKinds = [RefKind.ContainKind,RefKind.AliasTo,RefKind.HasambiguousKind,RefKind.Annotate,RefKind.ImportKind]

class EntKind(str, Enum):
    #function
    Function = "Function"
    #scope
    Module = "Module"
    Class = "Class"
    #Vars
    Variable = "Variable"
    ClassAttr = "Attribute"
    
    #external
    UnknownVar = "Unknown Variable"
    UnknownModule = "Unknown Module"
    
    #unused
    Alias = "Alias"
    Parameter = "Parameter"
    Anonymous = "Anonymous"
    ReferencedAttr = "Referenced Attribute"
    ModuleAlias = "Module Alias"
    UnresolvedAttr = "Unresolved Attribute"
    AmbiguousAttr = "Ambiguous Attribute"
    #package
    Package = "Package"
    #lambda
    AnonymousFunction = "AnonymousFunction"
    LambdaParameter = "LambdaParameter"

Scope_EntKinds = [EntKind.Function, EntKind.Module, EntKind.Class]
Internal_EntKinds = [EntKind.Variable, EntKind.ClassAttr, EntKind.Function, EntKind.Class, EntKind.UnresolvedAttr]
Var_EntKinds = [EntKind.Variable, EntKind.ClassAttr]
Ignored_EntKinds = [
                    EntKind.Alias, 
                    EntKind.UnknownVar, 
                    EntKind.UnknownModule,
                    EntKind.Parameter, 
                   EntKind.Anonymous, 
                   EntKind.ReferencedAttr, 
                   EntKind.ModuleAlias, 
                   EntKind.AmbiguousAttr, 
                   EntKind.Package, 
                   EntKind.AnonymousFunction,
                   EntKind.LambdaParameter] 

# KindSet is a kind for `Set` relation
# like
#
# def fun():
#     a = b
#
# Then fun set `Variable`(Entity) a
