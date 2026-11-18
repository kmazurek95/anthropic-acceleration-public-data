import sys, zipfile, hashlib
a, b = zipfile.ZipFile(sys.argv[1]), zipfile.ZipFile(sys.argv[2])
ia = {i.filename: i for i in a.infolist()}; ib = {i.filename: i for i in b.infolist()}
print("members direct/archived:", len(ia), len(ib))
print("only direct:", sorted(set(ia)-set(ib))); print("only archived:", sorted(set(ib)-set(ia)))
same = diff = 0
for n in sorted(set(ia)&set(ib)):
    da, db = a.read(n), b.read(n)
    ha, hb = hashlib.sha256(da).hexdigest(), hashlib.sha256(db).hexdigest()
    meta = (ia[n].date_time, ib[n].date_time, ia[n].compress_size, ib[n].compress_size)
    if ha == hb:
        same += 1
        if ia[n].date_time != ib[n].date_time or ia[n].compress_size != ib[n].compress_size:
            print("SAME content, metadata differs:", n, meta)
    else:
        diff += 1; print("CONTENT DIFFERS:", n, len(da), len(db), meta)
print("identical content:", same, "differing:", diff)
print("comment direct/archived:", a.comment, b.comment)
