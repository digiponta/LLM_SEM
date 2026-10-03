def decide(source_top,candidate_top,expected,margin,gain,global_pass=True):
    strict=candidate_top==expected and margin>=0.02
    degraded=source_top==expected and candidate_top==expected and margin>=0.01
    recovered=source_top!=expected and candidate_top==expected and margin>=0.005 and gain>0
    ok=global_pass and (strict or degraded or recovered)
    mode="RETENTION" if strict else ("DEGRADED" if degraded else ("RECOVERY" if recovered else "FAIL"))
    return ok,mode

def main():
    cases=[
        ("observed",("computer","computer","computer",0.012957,-0.042809),True,"DEGRADED"),
        ("strict",("computer","computer","computer",0.025,-0.05),True,"RETENTION"),
        ("weak",("computer","computer","computer",0.009,-0.01),False,"FAIL"),
        ("recovery",("computer","science","science",0.010,0.01),True,"RECOVERY"),
    ]
    failed=0
    for name,args,exp_ok,exp_mode in cases:
        ok,mode=decide(*args)
        good=(ok==exp_ok and mode==exp_mode)
        failed+=int(not good)
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {mode}")
    print("RESULT:","PASS" if failed==0 else "FAIL")
    raise SystemExit(1 if failed else 0)

if __name__=="__main__":
    main()
