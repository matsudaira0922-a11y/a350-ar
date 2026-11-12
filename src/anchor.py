from pxr import Sdf
def apply(prim):
    """Preliminary_AnchoringAPI on the root-level prim: first horizontal plane (Apple USD schema)."""
    prim.SetMetadata('apiSchemas', Sdf.TokenListOp.Create(prependedItems=['Preliminary_AnchoringAPI']))
    a=prim.CreateAttribute('preliminary:anchoring:type',Sdf.ValueTypeNames.Token,False,Sdf.VariabilityUniform); a.Set('plane')
    b=prim.CreateAttribute('preliminary:planeAnchoring:alignment',Sdf.ValueTypeNames.Token,False,Sdf.VariabilityUniform); b.Set('horizontal')
