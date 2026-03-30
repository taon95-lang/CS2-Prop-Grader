f = open('.replit').read()
old = '[deployment]\nrouter = "application"\ndeploymentTarget = "autoscale"\n\n[deployment.postBuild]\nargs = ["pnpm", "store", "prune"]\nenv = { "CI" = "true" }'
new = '[deployment]\nrun = ["sh", "-c", "cd cs2-bot && python bot.py"]\ndeploymentTarget = "reserved_vm"'
if old in f:
    open('.replit', 'w').write(f.replace(old, new))
    print("Done! Deployment configured for Reserved VM.")
else:
    print("Already updated or structure differs.")
    print(f[f.find('[deployment]'):f.find('[deployment]')+200])
